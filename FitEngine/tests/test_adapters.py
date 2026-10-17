"""Adapter from the avatar module (Baraah) to fit_engine's body document.

The response fixture is a real /generate-avatar response (see its "_source").
Its generated mesh drifted from the request (chest 101.6 cm generated vs 90
requested), which is exactly why the adapter must use the requested values.
"""

import http.client
import json
import os
import threading
import unittest

from fit_engine import recommend_size
from fit_engine.adapters import avatar_to_body
from fit_engine.server import make_server

from .helpers import comp, dress

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "avatar_response_female.json")


def load_response():
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


class TestAvatarResponse(unittest.TestCase):
    def setUp(self):
        self.resp = load_response()
        self.body, self.notes = avatar_to_body(self.resp)

    def test_uses_requested_not_generated(self):
        m = self.body["measurements"]
        self.assertEqual(m, {"chest": 90.0, "waist": 72.0, "hip": 96.0, "height": 200.0})
        # the generated mesh is far from the request; it must not leak in
        self.assertGreater(self.resp["measurements"]["chest_m"] * 100, 100)

    def test_weight_and_joint_lengths_not_used(self):
        m = self.body["measurements"]
        for key in ("weight", "shoulder_width", "arm_length", "inseam", "thigh"):
            self.assertNotIn(key, m)
        self.assertTrue(any("weight ignored" in n for n in self.notes))

    def test_metadata(self):
        self.assertEqual(self.body["source"], "avatar")
        self.assertEqual(self.body["user_id"], self.resp["avatar_id"])

    def test_end_to_end_recommendation(self):
        # dress target (regular, non-stretch): chest +4..+12, waist +3..+12, hip +4..+14
        # size M: chest 96-90=+6, waist 80-72=+8, hip 102-96=+6 -> all good
        r = recommend_size(self.body, dress())
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["recommended_size"], "M")
        self.assertEqual(comp(r, "chest")["body_cm"], 90.0)
        self.assertEqual(r["body_source"], "avatar")


class TestAvatarRequest(unittest.TestCase):
    def test_request_fields(self):
        body, _ = avatar_to_body({"height": 165, "weight": 60, "waist": 72, "chest": 90, "hip": 96,
                                  "gender": "female", "skin_tone": "#c8997e"})
        self.assertEqual(body["measurements"],
                         {"chest": 90, "waist": 72, "hip": 96, "height": 165})

    def test_male_shoulder_maps_to_shoulder_width(self):
        body, _ = avatar_to_body({"chest": 100, "waist": 85, "hip": 98, "shoulder": 46, "gender": "male"})
        self.assertEqual(body["measurements"]["shoulder_width"], 46)

    def test_bad_values_are_left_out_not_invented(self):
        body, _ = avatar_to_body({"chest": "90", "waist": True, "hip": None})
        self.assertEqual(body["measurements"], {})
        r = recommend_size(body, dress())
        self.assertEqual(r["status"], "invalid_input")

    def test_not_a_dict(self):
        body, notes = avatar_to_body([1, 2])
        self.assertEqual(body["measurements"], {})
        self.assertTrue(notes)


class TestAvatarToBodyEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = make_server("127.0.0.1", 0)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _post(self, payload):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        data = json.dumps(payload).encode()
        c.request("POST", "/avatar-to-body", body=data, headers={"Content-Type": "application/json"})
        r = c.getresponse()
        return r.status, json.loads(r.read())

    def test_response_converted(self):
        status, data = self._post({"avatar": load_response()})
        self.assertEqual(status, 200)
        self.assertEqual(data["body"]["measurements"]["chest"], 90.0)

    def test_bad_shape(self):
        status, _ = self._post({"nothing": 1})
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
