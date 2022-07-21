#ifndef OSC_STEP_H
#define OSC_STEP_H

#include <pseudo_inversion.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>

#include <Eigen/Dense>
#include <vector>
#include <fstream>


Eigen::Matrix<double, 7, 1> saturateTorqueRate(
    const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
    const Eigen::Matrix<double, 7, 1>& tau_J_d,
    const double delta_tau_max_) { 
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] =
        tau_J_d[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);    
  }
  return tau_d_saturated;
}



Eigen::Matrix<double, 7, 1> step(Eigen::Affine3d transform, 
                                Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian,
                                Eigen::Map<Eigen::Matrix<double, 7, 7>> mass,
                                Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis,
                                Eigen::Map<Eigen::Matrix<double, 7, 1>> q,
                                Eigen::Map<Eigen::Matrix<double, 7, 1>> dq,
                                Eigen::VectorXd q_d_nullspace_,
                                Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d,
                                Eigen::Vector3d position_d_,
                                Eigen::Vector3d velocity_d_,
                                Eigen::MatrixXd orientation_d_,
                                double delta_tau_max_,
                                Eigen::Matrix<double, 3, 3> kp_pos,
                                Eigen::Matrix<double, 3, 3> kp_rot,
                                Eigen::Matrix<double, 3, 3> kv_pos,
                                Eigen::Matrix<double, 3, 3> kv_rot,
                                bool train = false
                                ) {
    
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");
    double max_delta = delta_tau_max_;
    double joint_kp = 20.0;
    double joint_kv = sqrt(joint_kp) * 2;
    Eigen::Vector3d ee_position(transform.translation());
    Eigen::MatrixXd ee_rot(transform.linear());
    Eigen::Matrix<double, 3, 7> jacp(jacobian.block<3,7>(0,0));
    Eigen::Matrix<double, 3, 7> jacr(jacobian.block<3,7>(3,0));
    Eigen::Matrix<double, 3, 1> ee_vel(jacp * dq);
    Eigen::Matrix<double, 3, 1> ee_rot_vel(jacr * dq);
    Eigen::Matrix<double, 3, 1> orientation_velocity_error(-ee_rot_vel);

    Eigen::Matrix<double, 1, 3> rc1(ee_rot.block<3,1>(0,0));
    Eigen::Matrix<double, 1, 3> rc2(ee_rot.block<3,1>(0,1));
    Eigen::Matrix<double, 1, 3> rc3(ee_rot.block<3,1>(0,2));
    Eigen::Matrix<double, 1, 3> rd1(orientation_d_.block<3,1>(0,0));
    Eigen::Matrix<double, 1, 3> rd2(orientation_d_.block<3,1>(0,1));
    Eigen::Matrix<double, 1, 3> rd3(orientation_d_.block<3,1>(0,2));

    Eigen::Vector3d velocity_error = velocity_d_ -ee_vel;
    Eigen::Vector3d position_error = position_d_ - ee_position;

    Eigen::Vector3d orientation_error = 0.5 * (rc1.cross(rd1) + rc2.cross(rd2) + rc3.cross(rd3));
    Eigen::VectorXd desired_force = kp_pos * position_error + kv_pos * velocity_error;
    Eigen::VectorXd desired_torque = kp_rot * orientation_error + kv_rot * orientation_velocity_error;

    Eigen::MatrixXd lambda_inverse((jacobian * mass.inverse()) * jacobian.transpose());
    Eigen::MatrixXd lambda_pos_inverse((jacp * mass.inverse()) * jacp.transpose());
    Eigen::MatrixXd lambda_ori_inverse((jacr * mass.inverse()) * jacr.transpose());

    Eigen::MatrixXd lambda;
    Eigen::MatrixXd lambda_pos;
    Eigen::MatrixXd lambda_ori;
    osc::pseudoInverse(lambda_inverse, lambda);
    osc::pseudoInverse(lambda_pos_inverse, lambda_pos);
    osc::pseudoInverse(lambda_ori_inverse, lambda_ori);

    Eigen::Vector3d force = lambda_pos * desired_force;
    Eigen::Vector3d torque = lambda_ori * desired_torque;

    Eigen::Matrix<double, 6, 1> wrench;
    wrench.head(3) << force;
    wrench.tail(3) << torque;
    
    Eigen::Matrix<double, 7, 1> ctrl_torques(jacobian.transpose() * wrench);

    Eigen::MatrixXd jbar((mass.inverse() * jacobian.transpose()) * lambda);
    Eigen::MatrixXd nullspace_matrix(Eigen::MatrixXd::Identity(7, 7) - jbar * jacobian);
    Eigen::MatrixXd pose_torques(mass * (joint_kp * (q_d_nullspace_ - q) - joint_kv * dq));
    Eigen::MatrixXd nullspace_torques(nullspace_matrix.transpose() * pose_torques);
    
    Eigen::Matrix<double, 7, 1> torques;
    torques << ctrl_torques + nullspace_torques + coriolis;
    torques << saturateTorqueRate(torques, tau_J_d, max_delta);

    return torques;
};


#endif