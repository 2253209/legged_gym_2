import math
import select
import threading
import time
import numpy as np
from deploy.lcm_types.state_estimator_lcmt import state_estimator_lcmt
from deploy.lcm_types.leg_control_data_lcmt import leg_control_data_lcmt

class StateEstimator:
    def __init__(self, lc):
        self.lc = lc

        self.joint_pos = np.zeros(12)
        self.joint_vel = np.zeros(12)
        self.quat = np.zeros(4)
        self.omegaBody = np.zeros(3)

        self.state_subscription = self.lc.subscribe("state_estimator", self._state_cb)
        self.leg_subscription = self.lc.subscribe("leg_control_data_RL", self._leg_cb)
        self.running = True


    def _state_cb(self, channel, data):
        # print("update state")
        msg = state_estimator_lcmt.decode(data)
        self.quat = np.array(msg.quat)
        self.omegaBody = np.array(msg.omegaBody)
        # print(f"update imudata {msg.id}")

    def _leg_cb(self, channel, data):
        # print("update leg")
        msg = leg_control_data_lcmt.decode(data)
        self.joint_pos = np.array(msg.q)
        self.joint_vel = np.array(msg.qd)
        # print(f"update legdata {msg.id}")

    def poll(self, cb=None):
        t = time.time()
        try:
            timeout = 0.002
            while self.running:
                rfds, wfds, efds = select.select([self.lc.fileno()], [], [], timeout)
                if rfds:
                    # print("message received!")
                    self.lc.handle()
                    # print(f'Freq {1. / (time.time() - t)} Hz'); t = time.time()
                else:
                    continue
                    # print(f'waiting for message... Freq {1. / (time.time() - t)} Hz'); t = time.time()
                #    if cb is not None:
                #        cb()
        except KeyboardInterrupt:
            pass

    def spin(self):
        self.run_thread = threading.Thread(target=self.poll, daemon=True)
        self.run_thread.start()

    def close(self):
        self.lc.unsubscribe(self.state_subscription)
        self.running = False


if __name__ == "__main__":
    import lcm

    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
    se = StateEstimator(lc)
    try:
        se.spin()
    except KeyboardInterrupt:
        pass
    finally:
        se.close()

