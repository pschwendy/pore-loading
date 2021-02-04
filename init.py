from project import Project, dict_product
import numpy as np


project = Project()
replicas = [0]
phis = [0.69]
gars = [1.0] #, 0.5]
hgars = [0.4, 0.5, 0.6]
fugacities = np.linspace(1.0, 500, 8)
params = {
    'replica': replicas,
    'phi': phis,
    'guest_aspect_ratio': gars,
    'host_guest_area_ratio': hgars,
    'fugacity': fugacities,
    'sigma': [0.0],
    'lambdasigma': [0.1],
    }
for sp in dict_product(params):
    sp['n_guest_edges'] = 4
    sp['n_edges'] = 3
    sp['kT'] = 0.2
    sp['initial_config'] = 's3'
    sp['n_repeats'] = [16, 16]
    sp['patch_offset'] = 0.35
    job = project.open_job(sp).init()
    if job:
        t_eq = 1e6 * job.sp.fugacity + 2.2e6
        job.doc.stop_after = min(int(10e6), int(t_eq + 1e6))
        print('Added job with id {}'.format(job._id[:6]))

project.run(names=['initialize'], progress=True)
