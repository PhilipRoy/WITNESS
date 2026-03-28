# W.I.T.N.E.S.S. - Web-based Interrogation and Testimony via a Neural Engaged Speech System
# Copyright (C) 2026 Philip Roy <https://www.bluengrey.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Shared utility functions for the WITNESS system.

Persona files are the core data structure that drives everything: they define
who the interviewee is, what they know, and how they should behave during an interview.
"""

import json

def load_persona(file):
    """Load and return a persona JSON from an open file object."""
    return json.load(file)

def save_persona(data, filename):
    """Write a persona dict to disk as formatted JSON."""
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
