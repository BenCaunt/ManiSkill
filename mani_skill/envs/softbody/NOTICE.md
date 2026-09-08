# Legacy soft-body code

`fill.py` and `geometry.py` adapt the task equations, initialization, and SDF
algorithm from ManiSkill 2 v0.5.3, commit
`493be36121a9dd06071a57172274babe617b789f`, by the ManiSkill authors:

- https://github.com/mani-skill/ManiSkill/blob/493be36121a9dd06071a57172274babe617b789f/mani_skill2/envs/mpm/fill_env.py
- https://github.com/mani-skill/ManiSkill/blob/493be36121a9dd06071a57172274babe617b789f/mani_skill2/envs/mpm/utils.py

The original README places soft-body environments under the NVIDIA Source Code
License for Warp. These adaptations retain those terms, including the research
or evaluation use restriction. The complete license is preserved at
[`warp_maniskill/LICENSE.md`](../../../warp_maniskill/LICENSE.md).
The repository-wide Apache license does not replace those legacy terms.

The modifications replace SAPIEN 2 interfaces with ManiSkill 3/SAPIEN 3 scene,
controller, observation, and rendering APIs. The numerical SDF sampling retains
the original face-normal convention. Recorded bucket mass/inertia are reference
simulation estimates, not physical measurements.

Legacy meshes and textures are supplied separately and are not redistributed
in this directory. The original asset bundle has separate CC-BY-NC-4.0 terms.
