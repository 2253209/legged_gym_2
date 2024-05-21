"""LCM type definitions
This file automatically generated by lcm.
DO NOT MODIFY BY HAND!!!!
"""

from io import BytesIO
import struct

class key_command_lcmt(object):

    __slots__ = ["use_tau_mapping_rl", "use_ankle_motor_kd_rl", "timestamp_us", "id", "robot_id"]

    __typenames__ = ["int16_t", "int16_t", "int64_t", "int64_t", "int64_t"]

    __dimensions__ = [None, None, None, None, None]

    def __init__(self):
        self.use_tau_mapping_rl = 0
        """ LCM Type: int16_t """
        self.use_ankle_motor_kd_rl = 0
        """ LCM Type: int16_t """
        self.timestamp_us = 0
        """ LCM Type: int64_t """
        self.id = 0
        """ LCM Type: int64_t """
        self.robot_id = 0
        """ LCM Type: int64_t """

    def encode(self):
        buf = BytesIO()
        buf.write(key_command_lcmt._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(struct.pack(">hhqqq", self.use_tau_mapping_rl, self.use_ankle_motor_kd_rl, self.timestamp_us, self.id, self.robot_id))

    @staticmethod
    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != key_command_lcmt._get_packed_fingerprint():
            raise ValueError("Decode error")
        return key_command_lcmt._decode_one(buf)

    @staticmethod
    def _decode_one(buf):
        self = key_command_lcmt()
        self.use_tau_mapping_rl, self.use_ankle_motor_kd_rl, self.timestamp_us, self.id, self.robot_id = struct.unpack(">hhqqq", buf.read(28))
        return self

    @staticmethod
    def _get_hash_recursive(parents):
        if key_command_lcmt in parents: return 0
        tmphash = (0x3d45a7605aa27fc0) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff) + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _packed_fingerprint = None

    @staticmethod
    def _get_packed_fingerprint():
        if key_command_lcmt._packed_fingerprint is None:
            key_command_lcmt._packed_fingerprint = struct.pack(">Q", key_command_lcmt._get_hash_recursive([]))
        return key_command_lcmt._packed_fingerprint

    def get_hash(self):
        """Get the LCM hash of the struct"""
        return struct.unpack(">Q", key_command_lcmt._get_packed_fingerprint())[0]

