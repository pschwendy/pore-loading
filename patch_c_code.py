code_patch_SQWELL = """

// initialize the patches
const vec3<float> verts[] = {{
    {patch_locations}
    }};

// initialize pair energy
float pair_eng = 0.0;

// set the square well parameters and number of patches
const float epsilon = {epsilon:.15f};
const float sigma = {sigma:.15f};
const float lambdasigma = {lambdasigma:.15f};
const int n_patches = {n_patches};
const unsigned int host_type = {host_type};

if (type_i != host_type || type_j != host_type)
{{
    return 0.0;
}}


// check patch overlaps
for (int patch_idx_on_i = 0; patch_idx_on_i < n_patches; ++patch_idx_on_i)
    {{
      // let r_m be the location of the patch on particle i
      vec3<float> r_m = rotate(q_i, verts[patch_idx_on_i]);

      // for each of these patches on particle i, loop through patches on j
      for (int patch_idx_on_j = 0; patch_idx_on_j < n_patches; ++patch_idx_on_j)
      {{
        // let r_n be the location of the patch on particle j
        vec3<float> r_n = rotate(q_j, verts[patch_idx_on_j]) + r_ij;

        // now the vector from r_m to r_n is just r_n - r_m
        // call that vector dr
        vec3<float> dr = r_n - r_m;

        // now check to see if the length of dr is less than the range of the square well
        float rsq = dot(dr, dr);
        if (rsq <= lambdasigma*lambdasigma)  // assumes sigma = 0
        {{
        pair_eng += -1.0;
        }}
      }}
    }}

// multiply the pair counts by epsilon and return energy
pair_eng = epsilon*pair_eng;
return pair_eng;
"""
