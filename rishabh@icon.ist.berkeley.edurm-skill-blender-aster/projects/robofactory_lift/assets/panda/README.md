# Panda asset subset

Source and revision are recorded in `SOURCE.json`; `LICENSE` is the upstream
Apache 2.0 license. `panda.xml` is a collision-only derivative with eleven
arm/hand collision meshes and primitive finger geometry. Unused visual assets,
actuators, tendons, equalities, and keyframes have been removed. The model is
directly loadable; joint frames and inertial properties are preserved.

`skill_composer.sim` instantiates two copies, enables gravity compensation, adds
fixed position controllers, and creates TCP sites and the proxy task geometry.
The original mesh files are unchanged.
