cmake_minimum_required(VERSION 2.8.3)
project(rtreach)

find_package(catkin REQUIRED COMPONENTS
  gazebo_ros
  roscpp
  rospy
  std_msgs
  sensor_msgs
  vandy_bluerov
  message_generation
  geometry_msgs
  cmake_modules
  tf
)

include_directories(
  ${catkin_INCLUDE_DIRS}

)

add_service_files(
   FILES
   obstacle_list.srv
)

add_message_files(
  FILES
  velocity_msg.msg
  angle_msg.msg
  stamped_ttc.msg
  interval.msg
  reach_tube.msg
  FloatStamped.msg
)

generate_messages(
   DEPENDENCIES
   sensor_msgs
   std_msgs
   geometry_msgs
 )

catkin_package(
  CATKIN_DEPENDS roscpp rospy sensor_msgs vandy_bluerov std_msgs geometry_msgs gazebo_ros cmake_modules
)

include_directories(
  reach_src
  ${catkin_INCLUDE_DIRS}
)

add_library(Rtreachvisrov
            reach_src/face_lift_uuv_visualization.c 
            reach_src/uuv_model_vis.c 
            reach_src/dynamics_blue_rov.c 
            reach_src/geometry.c
            reach_src/interval.c  
            reach_src/simulate_uuv.c 
            reach_src/util.c 
            reach_src/uuv_safety.c)
target_compile_definitions(Rtreachvisrov PUBLIC UUV_MODEL)


add_library(Rtreachdynrov 
            reach_src/face_lift_uuv_dynamic.c 
            reach_src/dynamics_blue_rov.c  
            reach_src/uuv_model_dynamic.c 
            reach_src/geometry.c 
            reach_src/interval.c  
            reach_src/simulate_uuv.c 
            reach_src/util.c)
target_compile_definitions(Rtreachdynrov PUBLIC UUV_MODEL)

add_library(Rtreachvis 
            reach_src/face_lift_uuv_visualization.c 
            reach_src/uuv_model_vis.c 
            reach_src/dynamics_uuv_model.c 
            reach_src/geometry.c 
            reach_src/interval.c  
            reach_src/simulate_uuv.c 
            reach_src/util.c 
            reach_src/uuv_safety.c)
target_compile_definitions(Rtreachvis PUBLIC UUV_MODEL)

add_library(Rtreachdyn 
            reach_src/face_lift_uuv_dynamic.c 
            reach_src/dynamics_uuv_model.c  
            reach_src/uuv_model_dynamic.c 
            reach_src/geometry.c reach_src/interval.c  
            reach_src/simulate_uuv.c 
            reach_src/util.c)
target_compile_definitions(Rtreachdyn PUBLIC UUV_MODEL)

add_library(RtreachObs 
            reach_src/face_lift_obstacle_visualization.c 
            reach_src/dynamics_obstacle.c 
            reach_src/geometry.c 
            reach_src/interval.c  
            reach_src/simulate_obstacle.c 
            reach_src/util.c 
            reach_src/obstacle_model_plots.c)
target_compile_definitions(RtreachObs  PUBLIC OBSTACLE_MODEL)


add_library(RtreachUncertain SHARED
            reach_src/face_lifting_bicycle_uncertainty.c
            reach_src/dynamics_bicycle_uncertainty.c 
            reach_src/geometry.c 
            reach_src/interval.c 
            reach_src/util.c 
            reach_src/bicycle_model_uncertainty.c)
target_compile_definitions(RtreachUncertain PUBLIC BICYCLE_MODEL)

add_library(RtreachUncertainROV SHARED
            reach_src/face_lifting_bicycle_uncertainty.c
            reach_src/dynamics_bicycle_uncertainty_rov.c 
            reach_src/geometry.c 
            reach_src/interval.c 
            reach_src/util.c 
            reach_src/bicycle_model_uncertainty_rov.c)
target_compile_definitions(RtreachUncertainROV PUBLIC BICYCLE_MODEL)


add_library(Rtreachdynbc 
            reach_src/bicycle_model_disturbance.c 
            reach_src/face_lift_disturbance.c 
            reach_src/dynamics_bicycle_model_disturbance.c 
            reach_src/face_lift_parametrizeable.c 
            reach_src/dynamics_bicycle_model.c 
            reach_src/bicycle_model_parametrizeable.c  
            reach_src/face_lift_degraded.c  
            reach_src/dynamics_bicycle_model_degraded.c 
            reach_src/bicycle_model_degraded.c 
            reach_src/geometry.c 
            reach_src/interval.c 
            reach_src/util.c 
            reach_src/simulate_bicycle.c)
target_compile_definitions(Rtreachdynbc PUBLIC BICYCLE_MODEL)

add_library(Rtreachdynbciver 
            reach_src/bicycle_model_disturbance.c 
            reach_src/face_lift_disturbance.c 
            reach_src/dynamics_bicycle_model_disturbance.c 
            reach_src/face_lift_parametrizeable.c 
            reach_src/dynamics_bicycle_model_iver.c 
            reach_src/bicycle_model_parametrizeable.c  
            reach_src/face_lift_degraded.c  
            reach_src/dynamics_bicycle_model_degraded.c 
            reach_src/bicycle_model_degraded.c 
            reach_src/geometry.c 
            reach_src/interval.c 
            reach_src/util.c 
            reach_src/simulate_bicycle.c)
target_compile_definitions(Rtreachdynbciver PUBLIC BICYCLE_MODEL)

add_executable(visualize_node src/visualize_reachset.cpp)
add_dependencies(visualize_node ${catkin_EXPORTED_TARGETS})
add_dependencies(visualize_node rtreach_generate_messages_cpp)
set_target_properties(visualize_node PROPERTIES COMPILE_FLAGS "-DUUV_MODEL")
target_link_libraries(visualize_node ${catkin_LIBRARIES} Rtreachvis)


add_executable(visualize_node_rov src/visualize_reachset_blue_rov.cpp)
add_dependencies(visualize_node_rov ${catkin_EXPORTED_TARGETS})
add_dependencies(visualize_node_rov rtreach_generate_messages_cpp)
set_target_properties(visualize_node_rov PROPERTIES COMPILE_FLAGS "-DUUV_MODEL")
target_link_libraries(visualize_node_rov ${catkin_LIBRARIES} Rtreachvisrov)

add_executable(visualize_node_rov_bicycle src/visualize_reachset_blue_rov_bicycle.cpp)
add_dependencies(visualize_node_rov_bicycle ${catkin_EXPORTED_TARGETS})
add_dependencies(visualize_node_rov_bicycle rtreach_generate_messages_cpp)
set_target_properties(visualize_node_rov_bicycle PROPERTIES COMPILE_FLAGS "-DBICYCLE_MODEL")
target_link_libraries(visualize_node_rov_bicycle ${catkin_LIBRARIES} Rtreachdynbc)


add_executable(visualize_node_bicycle_iver src/visualize_reachset_iver_bicycle.cpp)
add_dependencies(visualize_node_bicycle_iver ${catkin_EXPORTED_TARGETS})
add_dependencies(visualize_node_bicycle_iver rtreach_generate_messages_cpp)
set_target_properties(visualize_node_bicycle_iver PROPERTIES COMPILE_FLAGS "-DBICYCLE_MODEL")
target_link_libraries(visualize_node_bicycle_iver ${catkin_LIBRARIES} Rtreachdynbciver)

add_executable(visualize_obs src/visualize_obstacle.cpp)
add_dependencies(visualize_obs rtreach_generate_messages_cpp)
set_target_properties(visualize_obs PROPERTIES COMPILE_FLAGS "-DOBSTACLE_MODEL")
target_link_libraries(visualize_obs ${catkin_LIBRARIES} RtreachObs)


add_executable(reach_node src/reach_node_iver.cpp)
add_dependencies(reach_node rtreach_generate_messages_cpp)
add_dependencies(reach_node ${catkin_EXPORTED_TARGETS})
set_target_properties(reach_node  PROPERTIES COMPILE_FLAGS "-DBICYCLE_MODEL")
target_link_libraries(reach_node  ${catkin_LIBRARIES} Rtreachdynbciver )

add_executable(reach_node_rov src/reach_node_blue_rov.cpp)
add_dependencies(reach_node_rov rtreach_generate_messages_cpp)
add_dependencies(reach_node_rov ${catkin_EXPORTED_TARGETS})
set_target_properties(reach_node_rov  PROPERTIES COMPILE_FLAGS "-DUUV_MODEL")
target_link_libraries(reach_node_rov  ${catkin_LIBRARIES} Rtreachdynrov)

add_executable(reach_node_rov_bicycle src/reach_node_blue_rov_bicycle.cpp)
add_dependencies(reach_node_rov_bicycle rtreach_generate_messages_cpp)
add_dependencies(reach_node_rov_bicycle ${catkin_EXPORTED_TARGETS})
set_target_properties(reach_node_rov_bicycle  PROPERTIES COMPILE_FLAGS "-DBICYCLE_MODEL")
target_link_libraries(reach_node_rov_bicycle  ${catkin_LIBRARIES} Rtreachdynbc)

add_executable(reach_node_uncertain src/reach_node_uncertain.cpp)
add_dependencies(reach_node_uncertain rtreach_generate_messages_cpp)
add_dependencies(reach_node_uncertain ${catkin_EXPORTED_TARGETS})
set_target_properties(reach_node_uncertain  PROPERTIES COMPILE_FLAGS "-DBICYCLE_MODEL")
target_link_libraries(reach_node_uncertain  ${catkin_LIBRARIES} RtreachUncertain)

add_executable(reach_node_uncertain_rov src/reach_node_uncertain_rov.cpp)
add_dependencies(reach_node_uncertain_rov rtreach_generate_messages_cpp)
add_dependencies(reach_node_uncertain_rov ${catkin_EXPORTED_TARGETS})
set_target_properties(reach_node_uncertain_rov  PROPERTIES COMPILE_FLAGS "-DBICYCLE_MODEL")
target_link_libraries(reach_node_uncertain_rov  ${catkin_LIBRARIES} RtreachUncertainROV )
