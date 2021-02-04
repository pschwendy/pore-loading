import copy
import flow
from flow import FlowProject, directives, cmd
from flow import environments
import glob
import gsd.hoomd
import itertools
import math
import numpy as np
import os
import rowan

import patch_c_code


class Project(FlowProject):
    def __init__(self):
        flow.FlowProject.__init__(self)
        self.required_defaults = (
            ('gsd_frequency', int(1e4)),
            ('thermo_frequency', int(1e4)),
            ('n_tune_blocks', 5),
            ('n_tune_steps', 1000),
            ('n_steps_per_block', 50000),
            )

    @staticmethod
    def generate_patch_location_c_code(job):
        ret_str = ''
        for pl in job.doc.patch_locations:
            ret_str += 'vec3<float>({}),\n'.format(', '.join(map(str, pl)))
        return ret_str

    @staticmethod
    def s3_unitcell(f):
        """Open solid unit cell

        Args
        ----
        f : float
            The patch offset statepoint parameter

        Assumes f = ±1 corresponds to the patch on the vertex

        """
        import hoomd
        N = 2
        a1 = np.array([1, 0, 0])
        theta = np.deg2rad(60)
        a2 = np.array([np.cos(theta), np.sin(theta), 0])
        a3 = np.array([0, 0, 1])
        pos = [[0, 0, 0], 1/3*(a1+a2)]
        x = f * 90
        orientations = np.deg2rad(np.array([-30+x, 150+x]))
        orientations = [rowan.from_axis_angle(a3, t) for t in orientations]
        return hoomd.lattice.unitcell(N, a1, a2, a3, dimensions=2,
                position=pos, orientation=orientations)


@Project.operation
@Project.post.isfile('init.gsd')
def initialize(job):
    import hoomd
    import hoomd.hpmc
    import scipy.spatial
    """Sets up the system and sets default values for writers and stuf

    """
    # get sp params
    f = job.sp.patch_offset
    n_e = job.sp.n_edges
    n_ge = job.sp.n_guest_edges
    gar = job.sp.guest_aspect_ratio
    host_guest_area_ratio = job.sp.host_guest_area_ratio
    n_repeats = job.sp.n_repeats
    seed = job.sp.replica

    # initialize the hoomd context
    msg_fn = job.fn('init-hoomd.log')
    hoomd.context.initialize('--mode=cpu --msg-file={}'.format(msg_fn))

    # figure out shape vertices and patch locations
    xs = np.array([np.cos(n*2*np.pi/n_e) for n in range(n_e)])
    ys = np.array([np.sin(n*2*np.pi/n_e) for n in range(n_e)])
    zs = np.zeros_like(ys)
    vertices = np.vstack((xs, ys, zs)).T
    A_particle = scipy.spatial.ConvexHull(vertices[:, :2]).volume
    vertices = vertices - np.mean(vertices, axis=0)
    vertex_vertex_vectors = np.roll(vertices, -1, axis=0) - vertices
    half_edge_locations = vertices + 0.5 * vertex_vertex_vectors
    patch_locations = half_edge_locations + f * (vertices - half_edge_locations)

    # make the guest particles
    xs = np.array([np.cos(n*2*np.pi/n_ge) for n in range(n_ge)])
    ys = np.array([np.sin(n*2*np.pi/n_ge) for n in range(n_ge)])
    zs = np.zeros_like(ys)
    guest_vertices = np.vstack((xs, ys, zs)).T
    rot_quat = rowan.from_axis_angle([0, 0, 1], 2*np.pi/n_ge/2)
    guest_vertices = rowan.rotate(rot_quat, guest_vertices)
    guest_vertices[:, 0] *= gar
    target_guest_area = A_particle * host_guest_area_ratio
    current_guest_area = scipy.spatial.ConvexHull(guest_vertices[:, :2]).volume
    guest_vertices *= np.sqrt(target_guest_area / current_guest_area)


    # save everything into the job doc that we need to
    if hoomd.comm.get_rank() == 0:
        job.doc.vertices = vertices
        job.doc.patch_locations = patch_locations
        job.doc.A_particle = A_particle
        job.doc.guest_vertices = guest_vertices
    hoomd.comm.barrier()

    # build the system
    if job.sp.initial_config in ('open', 's3'):
        uc = job._project.s3_unitcell(job.sp.patch_offset)
        system = hoomd.init.create_lattice(uc, n_repeats)
    else:
        raise NotImplementedError('Initialization not implemented.')

    # restart writer; period=None since we'll just call write_restart() at end
    restart_writer = hoomd.dump.gsd(filename=job.fn('restart.gsd'),
            group=hoomd.group.all(), truncate=True, period=None,
            phase=0)

    # set up the integrator with the shape info
    mc = hoomd.hpmc.integrate.convex_polygon(seed=seed, d=0, a=0)
    mc.shape_param.set('A', vertices=vertices[:, :2])
    total_particle_area = len(system.particles) * A_particle
    phi = total_particle_area / system.box.get_volume()
    sf = np.sqrt(phi / job.sp.phi)
    hoomd.update.box_resize(
            Lx=system.box.Lx*sf,
            Ly=system.box.Ly*sf, 
            period=None,
            )
    restart_writer.dump_shape(mc)
    restart_writer.dump_state(mc)
    mc.set_params(d=0.1, a=0.5)

    # save everything into the job doc that we need to
    if hoomd.comm.get_rank() == 0:
        job.doc.mc_d = {x: 0.05 for x in system.particles.types}
        job.doc.mc_a = {x: 0.1 for x in system.particles.types}
        job.doc.vertices = vertices
        job.doc.patch_locations = patch_locations
        job.doc.A_particle = A_particle
        for k, v in job._project.required_defaults:
            job.doc.setdefault(k, v)
        os.system('cp {} {}'.format(job.fn('restart.gsd'), job.fn('init.gsd')))
    hoomd.comm.barrier()
    restart_writer.write_restart()
    if hoomd.comm.get_rank() == 0:
        os.system('cp {} {}'.format(job.fn('restart.gsd'), job.fn('init.gsd')))
    hoomd.comm.barrier()
    return


def done_running(job):
    ts = job.doc.get('timestep', None)
    sa = job.doc.get('stop_after', None)
    t0 = job.doc.get('ts_after_tuning', None)
    if any((ts is None, sa is None, t0 is None)):
        return False
    if job.doc.get('continue_running', None) == False:
        return False
    return job.doc.timestep > job.doc.stop_after + job.doc.ts_after_tuning


@Project.operation
@Project.pre.after(initialize)
@Project.post(done_running)
@directives(executable='singularity exec software-greatlakes.simg python3')
@directives(nranks=lambda job: job.doc.get('nranks', 4))
def sample(job):
    import hoomd
    import hoomd.hpmc
    import hoomd.jit

    # statepoint parameters
    seed = job.sp.replica
    kT = job.sp.kT
    fugacity = job.sp.fugacity

    # values from document
    gsd_frequency = job.doc.gsd_frequency
    thermo_frequency = job.doc.thermo_frequency
    n_tune_blocks = job.doc.n_tune_blocks
    n_tune_steps = job.doc.n_tune_steps
    mc_d, mc_a = job.doc.mc_d, job.doc.mc_a
    vertices = np.array(job.doc.vertices)[:, :2]
    guest_vertices = np.array(job.doc.guest_vertices)[:, :2]
    do_tuning = job.doc.get('do_tuning', True)
    n_steps_per_block = job.doc.n_steps_per_block

    # handle hoomd message files
    msg_fn = job.fn('hoomd-log.txt')
    hoomd.context.initialize('--mode=cpu')
    hoomd.option.set_msg_file(msg_fn)

    # initialize system from restart file
    initial_gsd_fn = job.fn('init.gsd')
    restart_fn = job.fn('restart.gsd')
    system = hoomd.init.read_gsd(initial_gsd_fn, restart_fn)

    gsd_fn = job.fn('traj.gsd')
    gsd_writer = hoomd.dump.gsd(gsd_fn, gsd_frequency,
            hoomd.group.all(), overwrite=False, truncate=False,
            dynamic=['attribute', 'momentum'])
    restart_frequency = gsd_frequency
    restart_writer = hoomd.dump.gsd(filename=restart_fn,
            group=hoomd.group.all(), truncate=True, period=restart_frequency,
            phase=0)
    mc = hoomd.hpmc.integrate.convex_polygon(seed=seed, restore_state=True)
    mc.shape_param.set('A', vertices=vertices)
    current_types = [x for x in system.particles.types]
    if 'B' not in current_types:
        system.particles.types.add('B')
    mc.shape_param.set('B', vertices=guest_vertices)
    mc_tuner = hoomd.hpmc.util.tune(mc, tunables=['d', 'a'], target=0.33)
    restart_writer.dump_state(mc)
    gsd_writer.dump_state(mc)
    restart_writer.dump_shape(mc)
    gsd_writer.dump_shape(mc)
    logger_info = {
            'thermo.txt': ['volume', 'lx', 'ly', 'lz', 'xy'], 
            'hpmc-stats.txt':
                ['hpmc_sweep', 'hpmc_translate_acceptance',
                 'hpmc_rotate_acceptance', 'hpmc_d', 'hpmc_a'],
            'hpmc-patch-stats.txt':
                ['hpmc_patch_energy', 'hpmc_patch_rcut'],
            'hpmc-muvt-stats.txt':
                ['hpmc_muvt_N_B'],
            }
    loggers = {}
    for fn, quantities in logger_info.items():
        loggers[fn[:-4]] = hoomd.analyze.log(job.fn(fn), quantities,
                thermo_frequency, header_prefix='# ', overwrite=False)

    # patches
    host_type = system.particles.pdata.getTypeByName('A')
    patch_code = patch_c_code.code_patch_SQWELL.format(
            patch_locations=job._project.generate_patch_location_c_code(job),
            n_patches=len(job.doc.patch_locations),
            sigma=job.sp.sigma,
            lambdasigma=job.sp.lambdasigma,
            host_type=host_type,
            epsilon=1/kT,
            )
    patches = hoomd.jit.patch.user(
            mc,
            code=patch_code,
            r_cut=2.0,
            array_size=0)

    # tune
    tuners = [mc_tuner]
    if do_tuning:
        for tune_block in range(n_tune_blocks):
            hoomd.run(n_tune_steps)
            for tuner in tuners:
                tuner.update()

    # make note of timestep after tuning
    if hoomd.comm.get_rank() == 0:
        job.doc.setdefault('ts_after_tuning', hoomd.get_step())
        job.doc.timestep = hoomd.get_step()
        job.doc.do_tuning = False
    hoomd.comm.barrier()

    # add the muvt updater
    muvt = hoomd.hpmc.update.muvt(mc, seed=seed, period=5, transfer_types=['B'])
    muvt.set_fugacity(type='B', fugacity=fugacity)

    # run
    keep_running = True
    while keep_running:
        if job.doc.timestep > job.doc.stop_after + job.doc.ts_after_tuning:
            keep_running = False
        try:
            hoomd.run(n_steps_per_block)
            if hoomd.comm.get_rank() == 0:
                job.doc.timestep = hoomd.get_step()
            hoomd.comm.barrier()
        except hoomd.WalltimeLimitReached:
            restart_writer.write_restart()
            if hoomd.comm.get_rank() == 0:
                job.doc.timestep = hoomd.get_step()
            hoomd.comm.barrier()
            return
    return


def dict_product(dd):
    keys = dd.keys()
    for element in itertools.product(*dd.values()):
        yield dict(zip(keys, element))


if __name__ == '__main__':
    pr = Project()
    output_dir = os.path.join(pr.root_directory(), 'output-files')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    pr.main()
