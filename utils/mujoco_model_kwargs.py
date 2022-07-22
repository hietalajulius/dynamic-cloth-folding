

model_kwarg_ranges = dict(
    joint_solimp_low=(0.98, 0.9999),
    joint_solimp_high=(0.98, 0.9999),
    joint_solimp_width=(0.01, 0.03),
    joint_solref_timeconst=(0.01, 0.05),
    joint_solref_dampratio=(0.98, 1.01555555555556),

    tendon_shear_solimp_low=(0.98, 0.9999),
    tendon_shear_solimp_high=(0.98, 0.9999),
    tendon_shear_solimp_width=(0.01, 0.03),
    tendon_shear_solref_timeconst=(0.01, 0.05),
    tendon_shear_solref_dampratio=(0.98, 1.01555555555556),

    tendon_main_solimp_low=(0.98, 0.9999),
    tendon_main_solimp_high=(0.98, 0.9999),
    tendon_main_solimp_width=(0.01, 0.03),
    tendon_main_solref_timeconst=(0.01, 0.05),
    tendon_main_solref_dampratio=(0.98, 1.01555555555556),

    geom_solimp_low=(0.98, 0.9999),
    geom_solimp_high=(0.98, 0.9999),
    geom_solimp_width=(0.01, 0.03),
    geom_solref_timeconst=(0.01, 0.05),
    geom_solref_dampratio=(0.98, 1.01555555555556),

    geom_size=(0.008, 0.011),
    friction=(0.01, 10),
    impratio=(1, 40)
)

model_kwarg_choices = dict(
    cone_type=["pyramidal", "elliptic"]
)

appearance_kwarg_ranges = dict(
    cloth_texture_width=(1, 1000),
    cloth_texture_height=(1, 1000),
    cloth_texture_random=(0, 1),
    cloth_texture_r_1=(0, 1),
    cloth_texture_g_1=(0, 1),
    cloth_texture_b_1=(0, 1),
    cloth_texture_r_2=(0, 1),
    cloth_texture_g_2=(0, 1),
    cloth_texture_b_2=(0, 1),
    cloth_texture_repeat_x=(1, 100),
    cloth_texture_repeat_y=(1, 100),

    table_texture_width=(1, 1000),
    table_texture_height=(1, 1000),
    table_texture_random=(0, 1),
    table_texture_r_1=(0, 0.9),
    table_texture_g_1=(0, 0.9),
    table_texture_b_1=(0, 0.9),
    table_texture_r_2=(0, 0.9),
    table_texture_g_2=(0, 0.9),
    table_texture_b_2=(0, 0.9),
    table_texture_repeat_x=(1, 100),
    table_texture_repeat_y=(1, 100),

    floor_texture_width=(1, 1000),
    floor_texture_height=(1, 1000),
    floor_texture_random=(0, 1),
    floor_texture_r_1=(0, 0.9),
    floor_texture_g_1=(0, 0.9),
    floor_texture_b_1=(0, 0.9),
    floor_texture_r_2=(0, 0.9),
    floor_texture_g_2=(0, 0.9),
    floor_texture_b_2=(0, 0.9),
    floor_texture_repeat_x=(1, 100),
    floor_texture_repeat_y=(1, 100),

    robot1_texture_width=(1, 1000),
    robot1_texture_height=(1, 1000),
    robot1_texture_random=(0, 1),
    robot1_texture_r_1=(0, 1),
    robot1_texture_g_1=(0, 1),
    robot1_texture_b_1=(0, 1),
    robot1_texture_r_2=(0, 1),
    robot1_texture_g_2=(0, 1),
    robot1_texture_b_2=(0, 1),
    robot1_texture_repeat_x=(1, 100),
    robot1_texture_repeat_y=(1, 100),

    robot2_texture_width=(1, 1000),
    robot2_texture_height=(1, 1000),
    robot2_texture_random=(0, 1),
    robot2_texture_r_1=(0, 1),
    robot2_texture_g_1=(0, 1),
    robot2_texture_b_1=(0, 1),
    robot2_texture_r_2=(0, 1),
    robot2_texture_g_2=(0, 1),
    robot2_texture_b_2=(0, 1),
    robot2_texture_repeat_x=(1, 100),
    robot2_texture_repeat_y=(1, 100),
)

appearance_kwarg_choices = dict(
    cloth_texture_builtin=["gradient", "checker", "flat"],
    cloth_texture_mark=["none", "edge", "cross", "random"],

    table_texture_builtin=["gradient", "checker", "flat"],
    table_texture_mark=["none", "edge", "cross", "random"],

    floor_texture_builtin=["gradient", "checker", "flat"],
    floor_texture_mark=["none", "edge", "cross", "random"],

    robot1_texture_builtin=["gradient", "checker", "flat"],
    robot1_texture_mark=["none", "edge", "cross", "random"],

    robot2_texture_builtin=["gradient", "checker", "flat"],
    robot2_texture_mark=["none", "edge", "cross", "random"],

)

BASE_MODEL_KWARGS = dict(
    joint_solimp_low=0.986633333333333,
    joint_solimp_high=0.9911,
    joint_solimp_width=0.03,
    joint_solref_timeconst=0.03,
    joint_solref_dampratio=1.01555555555556,

    tendon_shear_solimp_low=0.98,
    tendon_shear_solimp_high=0.99,
    tendon_shear_solimp_width=0.03,
    tendon_shear_solref_timeconst=0.05,
    tendon_shear_solref_dampratio=1.01555555555556,

    tendon_main_solimp_low=0.993266666666667,
    tendon_main_solimp_high=0.9966,
    tendon_main_solimp_width=0.004222222222222,
    tendon_main_solref_timeconst=0.01,
    tendon_main_solref_dampratio=0.98,

    geom_solimp_low=0.984422222222222,
    geom_solimp_high=0.9922,
    geom_solimp_width=0.007444444444444,
    geom_solref_timeconst=0.005,
    geom_solref_dampratio=1.01555555555556,

    geom_size=0.011,
    friction=0.05,
    impratio=20,
    cone_type="pyramidal",
)
