

model_kwarg_ranges = dict(
    joint_solimp_low = (0.98,0.9999),
    joint_solimp_high = (0.98,0.9999),
    joint_solimp_width = (0.01, 0.03),
    joint_solref_timeconst  = (0.01, 0.05),
    joint_solref_dampratio = (0.98, 1.01555555555556),

    tendon_shear_solimp_low = (0.98,0.9999),
    tendon_shear_solimp_high = (0.98,0.9999),
    tendon_shear_solimp_width = (0.01, 0.03),
    tendon_shear_solref_timeconst  = (0.01, 0.05),
    tendon_shear_solref_dampratio = (0.98, 1.01555555555556),

    tendon_main_solimp_low = (0.98,0.9999),
    tendon_main_solimp_high = (0.98,0.9999),
    tendon_main_solimp_width = (0.01, 0.03),
    tendon_main_solref_timeconst  = (0.01, 0.05),
    tendon_main_solref_dampratio = (0.98, 1.01555555555556),

    geom_solimp_low = (0.98,0.9999),
    geom_solimp_high = (0.98,0.9999),
    geom_solimp_width = (0.01, 0.03),
    geom_solref_timeconst  = (0.01, 0.05),
    geom_solref_dampratio = (0.98, 1.01555555555556),

    grasp_solimp_low = (0.99,0.9999),
    grasp_solimp_high = (0.99,0.9999),
    grasp_solimp_width = (0.01, 0.02),
    grasp_solref_timeconst  = (0.01, 0.02),
    grasp_solref_dampratio = (0.98, 1.01555555555556),

    geom_size = (0.008, 0.011),
    friction = (0.01, 10),
    impratio = (1, 40)
)

model_kwarg_choices = dict(
    cone_type = ["pyramidal", "elliptic"]
)

appearance_kwarg_ranges = dict(
    cloth_texture_width = (1,1000),
    cloth_texture_height = (1,1000),
    cloth_texture_random = (0,1),
    cloth_texture_r_1 = (0,1),
    cloth_texture_g_1 = (0,1),
    cloth_texture_b_1 = (0,1),
    cloth_texture_r_2 = (0,1),
    cloth_texture_g_2 = (0,1),
    cloth_texture_b_2 = (0,1),
    cloth_texture_repeat_x = (1,100),
    cloth_texture_repeat_y = (1,100),

    table_texture_width = (1,1000),
    table_texture_height = (1,1000),
    table_texture_random = (0,1),
    table_texture_r_1 = (0,0.9),
    table_texture_g_1 = (0,0.9),
    table_texture_b_1 = (0,0.9),
    table_texture_r_2 = (0,0.9),
    table_texture_g_2 = (0,0.9),
    table_texture_b_2 = (0,0.9),
    table_texture_repeat_x = (1,100),
    table_texture_repeat_y = (1,100),

    floor_texture_width = (1,1000),
    floor_texture_height = (1,1000),
    floor_texture_random = (0,1),
    floor_texture_r_1 = (0,0.9),
    floor_texture_g_1 = (0,0.9),
    floor_texture_b_1 = (0,0.9),
    floor_texture_r_2 = (0,0.9),
    floor_texture_g_2 = (0,0.9),
    floor_texture_b_2 = (0,0.9),
    floor_texture_repeat_x = (1,100),
    floor_texture_repeat_y = (1,100),

    robot1_texture_width = (1,1000),
    robot1_texture_height = (1,1000),
    robot1_texture_random = (0,1),
    robot1_texture_r_1 = (0,1),
    robot1_texture_g_1 = (0,1),
    robot1_texture_b_1 = (0,1),
    robot1_texture_r_2 = (0,1),
    robot1_texture_g_2 = (0,1),
    robot1_texture_b_2 = (0,1),
    robot1_texture_repeat_x = (1,100),
    robot1_texture_repeat_y = (1,100),

    robot2_texture_width = (1,1000),
    robot2_texture_height = (1,1000),
    robot2_texture_random = (0,1),
    robot2_texture_r_1 = (0,1),
    robot2_texture_g_1 = (0,1),
    robot2_texture_b_1 = (0,1),
    robot2_texture_r_2 = (0,1),
    robot2_texture_g_2 = (0,1),
    robot2_texture_b_2 = (0,1),
    robot2_texture_repeat_x = (1,100),
    robot2_texture_repeat_y = (1,100),
)

appearance_kwarg_choices = dict(
    cloth_texture_builtin = ["gradient", "checker", "flat"],
    cloth_texture_mark = ["none", "edge", "cross", "random"],

    table_texture_builtin = ["gradient", "checker", "flat"],
    table_texture_mark = ["none", "edge", "cross", "random"],

    floor_texture_builtin = ["gradient", "checker", "flat"],
    floor_texture_mark = ["none", "edge", "cross", "random"],

    robot1_texture_builtin = ["gradient", "checker", "flat"],
    robot1_texture_mark = ["none", "edge", "cross", "random"],

    robot2_texture_builtin = ["gradient", "checker", "flat"],
    robot2_texture_mark = ["none", "edge", "cross", "random"],

)

BATH_MODEL_KWARGS = dict(
    joint_solimp_low = 0.986633333333333,
    joint_solimp_high = 0.9911,
    joint_solimp_width = 0.03,
    joint_solref_timeconst  = 0.03,
    joint_solref_dampratio = 1.01555555555556,

    tendon_shear_solimp_low = 0.98,
    tendon_shear_solimp_high = 0.99,
    tendon_shear_solimp_width = 0.03,
    tendon_shear_solref_timeconst  = 0.05,
    tendon_shear_solref_dampratio = 1.01555555555556,

    tendon_main_solimp_low = 0.993266666666667,
    tendon_main_solimp_high = 0.9966,
    tendon_main_solimp_width = 0.004222222222222,
    tendon_main_solref_timeconst  = 0.01,
    tendon_main_solref_dampratio = 0.98,

    geom_solimp_low = 0.984422222222222,
    geom_solimp_high = 0.9922,
    geom_solimp_width = 0.007444444444444,
    geom_solref_timeconst  = 0.005,
    geom_solref_dampratio = 1.01555555555556,

    grasp_solimp_low = 0.99,
    grasp_solimp_high = 0.99,
    grasp_solimp_width = 0.01,
    grasp_solref_timeconst  = 0.01,
    grasp_solref_dampratio = 1,

    geom_size = 0.011,
    friction = 0.05,
    impratio = 20,
    cone_type = "pyramidal",
)

KITCHEN_MODEL_KWARGS = dict(
    geom_solimp_high=0.9995608287224704,
    geom_solimp_low=0.9926795889373894,
    geom_solimp_width=0.021690055982696063,
    geom_solref_dampratio=0.9811052070384793,
    geom_solref_timeconst=0.016690819023561253,

    grasp_solimp_high=0.9919910681148176,
    grasp_solimp_low=0.9988754968893396,
    grasp_solimp_width=0.017343662941659885,
    grasp_solref_dampratio=1.0079078957741223,
    grasp_solref_timeconst=0.01394900122154112,

    joint_solimp_high=0.9844183148525444,
    joint_solimp_low=0.9963514403005972,
    joint_solimp_width=0.02892870979618985,
    joint_solref_dampratio=1.0037779375661222,
    joint_solref_timeconst=0.047059398172076834,

    tendon_main_solimp_high=0.9934325621457002,
    tendon_main_solimp_low=0.9921426535371052,
    tendon_main_solimp_width=0.010633291641000795,
    tendon_main_solref_dampratio=0.9986721496048188,
    tendon_main_solref_timeconst=0.02610069486488337,

    tendon_shear_solimp_high=0.99077524912194,
    tendon_shear_solimp_low=0.9874698110831096,
    tendon_shear_solimp_width=0.02702580728824177,
    tendon_shear_solref_dampratio=0.9950335483736596,
    tendon_shear_solref_timeconst=0.010743309388481346,

    cone_type="pyramidal",
    friction=0.21836990104336332,
    impratio=8.040602647960528,
    geom_size=0.009953227930746909,
)

WIPE_MODEL_KWARGS = dict(
    geom_solimp_high=0.9944634963254524,
    geom_solimp_low=0.9950573135235632,
    geom_solimp_width=0.02264159172959327,
    geom_solref_dampratio=1.0109986493745688,
    geom_solref_timeconst=0.01034687833940208,

    grasp_solimp_high=0.994092995467093,
    grasp_solimp_low=0.9953859136739436,
    grasp_solimp_width=0.011539638652437229,
    grasp_solref_dampratio=1.0112437489275294,
    grasp_solref_timeconst=0.012827152841842491,
    
    joint_solimp_high=0.9924012012473988,
    joint_solimp_low=0.9950214140325514,
    joint_solimp_width=0.023872981622966846,
    joint_solref_dampratio=1.0101751047677787,
    joint_solref_timeconst=0.04240194570583116,

    tendon_main_solimp_high=0.9962886057237588,
    tendon_main_solimp_low=0.9989751339354273,
    tendon_main_solimp_width=0.027962193589671117,
    tendon_main_solref_dampratio=0.9934458109767637,
    tendon_main_solref_timeconst=0.04763463395564109,

    tendon_shear_solimp_high=0.9987644351800072,
    tendon_shear_solimp_low=0.9934485772738628,
    tendon_shear_solimp_width=0.011393145486057066,
    tendon_shear_solref_dampratio=0.9896011929885252,
    tendon_shear_solref_timeconst=0.01565016902244982,

    cone_type="elliptic",
    friction=0.5998064320505643,
    geom_size=0.009239293092755695,
    impratio=22.647202738935732,
)