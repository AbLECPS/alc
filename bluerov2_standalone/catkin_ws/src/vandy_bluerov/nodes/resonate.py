#!/usr/bin/env python
import rospy
import std_msgs.msg
import numpy as np

SIGMOID_LOWER_BOUND = 10**-15
SIGMOID_UPPER_BOUND = 1 - SIGMOID_LOWER_BOUND

SIGMOID_PARAMETERS = [0.18428081, 4.34419664]
SIGMOID_PARAMETERS_REALLOCATION = [5.73672305e+02, 7.07026933e-04]


class Resonate(object):
    """
    ROS node for the ReSonAte framework
    """
    def __init__(self):
        # Get any necessary parameters
        self._thruster_degradation_topic = rospy.get_param('~degradation_topic', "/uuv0/degradation_detector")
        self._thruster_reallocation_topic = rospy.get_param('~reallocation_topic', "/uuv0/thruster_reallocation")
        self._hazard_rate_topic = rospy.get_param('~hazard_rate_topic', "hazard_rate")

        # Init BTD class
        self._btd = BowTie()

        # Defub Hazard rate
        self._hazard_rate = None

        # Misc variables
        self._realloc_amount = None
        self.state = {"thruster_efficiency": 1.0, "reallocation": False}

        # Calculate initial hazard rate
        self._hazard_rate = self._btd.calc_hazard_rate(self.state)

        # Subscribe to thruster degradation and reallocation topics
        self._degradation_sub = rospy.Subscriber(self._thruster_degradation_topic, std_msgs.msg.Float32MultiArray,
                                                 self.degradation_cb, queue_size=1)

        # FIXME: Would be better to get reallocation matrix directly, but this is not currently published
        self._realloc_sub = rospy.Subscriber(self._thruster_reallocation_topic, std_msgs.msg.Float32MultiArray,
                                             self.realloc_cb, queue_size=1)

        # Setup publisher
        self._hazard_rate_pub = rospy.Publisher(self._hazard_rate_topic, std_msgs.msg.Float32, queue_size=1)

        # Publish hazard rate update
        msg = std_msgs.msg.Float32()        
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            if self._hazard_rate is not None:
                msg.data = self._hazard_rate
                self._hazard_rate_pub.publish(msg)
                print("ReSonAte hazard rate: "+str(self._hazard_rate))
            rate.sleep()

    def realloc_cb(self, msg):
        # [time, thr_id, efficiency]
        self._realloc_amount = 1.0 - msg.data[2]
        if self._realloc_amount > 0.01:
            self.state["reallocation"] = True
        else:
            self.state["reallocation"] = False

    def degradation_cb(self, msg):
        # [degraded_id, efficiency, prediction, credibility, confidence, decisions["snapshot"], decisions["comb"], am_output, softmax, combined_am_output]
        lec_prediction_class = msg.data[2]
        decision = msg.data[6]
        thruster_id = msg.data[0]
        thruster_efficiency = msg.data[1]

        #when AM confident and LEC output is degraded
        if (decision == 1 and lec_prediction_class < 22):                
            # Query the BTD to calculate hazard rate
            self.state["thruster_efficiency"] = thruster_efficiency
        self._hazard_rate = self._btd.calc_hazard_rate(self.state)



class BowTie(object):
    """Simple representation of Bow-Tie Diagram. Placeholder for more general solution"""
    def __init__(self):
        self.b1_sigmoid_params = SIGMOID_PARAMETERS
        self.b1_sigmoid_params_realloc = SIGMOID_PARAMETERS_REALLOCATION

    def rate_t1(self, state):
        return 1.0

    def prob_b1(self, state):
        # base_prob = 0.4
        # prob = base_prob

        # # For each relevant fault condition, if the fault is present multiply total probability by P(B1 | D) / P(B1)
        # # Otherwise, multiply by P(B1 | !D) / P(B1)
        # # See paper for complete explanation of equations
        # for detector_name, (p_true, p_false) in self.b1_cond_prob_table.items():
        #     if state["monitor_values"][detector_name]:
        #         prob *= p_true
        #     else:
        #         prob *= p_false
        #     prob /= base_prob

        # # Similarly for AM value, multiply total probability by P(B1 | AM) / P(B1)
        # log_martingale = state["monitor_values"]["lec_martingale"]
        # p_failure = bounded_sigmoid(log_martingale, *self.b1_sigmoid_parameters)
        # prob *= (1 - p_failure)
        # prob /= base_prob

        thruster_degradation = 1 - state["thruster_efficiency"]
        if state["reallocation"]:
            p_failure = bounded_sigmoid(thruster_degradation, *self.b1_sigmoid_params_realloc)
        else:
            p_failure = bounded_sigmoid(thruster_degradation, *self.b1_sigmoid_params)
        return 1 - p_failure

    def prob_b2(self, state):
        return 0.708333

    def calc_hazard_rate(self, state):
        # Calculate probabilities from bowtie
        # This will only work for this particular BTD. Need a generalized approach for a long-term solution
        r_top = self.rate_t1(state) * (1 - self.prob_b1(state))
        r_c1 = r_top * (1 - self.prob_b2(state))
        return r_c1


# Sigmoid function
def bounded_sigmoid(x, x0, k):
    y = 1 / (1 + np.exp(-k*(x-x0)))
    y = np.maximum(y, SIGMOID_LOWER_BOUND)
    y = np.minimum(y, SIGMOID_UPPER_BOUND)
    return y


if __name__ == '__main__':
    print('Starting ReSonAte node')
    rospy.init_node('resonate')
    try:
        node = Resonate()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
