#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for slurm_utils module."""

import os
from unittest.mock import patch

import pytest

from megatron.bridge.utils.slurm_utils import (
    _parse_slurm_nodelist,
    is_slurm_job,
    resolve_slurm_local_rank,
    resolve_slurm_master_addr,
    resolve_slurm_master_port,
    resolve_slurm_rank,
    resolve_slurm_world_size,
)


class TestIsSLURMJob:
    """Test is_slurm_job function."""

    @patch.dict(os.environ, {"SLURM_NTASKS": "8"})
    def test_is_slurm_job_true(self):
        """Test detection returns True when SLURM_NTASKS is set."""
        assert is_slurm_job() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_is_slurm_job_false(self):
        """Test detection returns False when SLURM_NTASKS is not set."""
        assert is_slurm_job() is False


class TestResolveSLURMRank:
    """Test resolve_slurm_rank function."""

    @patch.dict(os.environ, {"SLURM_NTASKS": "8", "SLURM_PROCID": "5"}, clear=True)
    def test_resolve_slurm_rank(self):
        """Test resolving rank from SLURM_PROCID."""
        assert resolve_slurm_rank() == 5

    @patch.dict(os.environ, {}, clear=True)
    def test_resolve_slurm_rank_not_slurm(self):
        """Test returns None when not in SLURM environment."""
        assert resolve_slurm_rank() is None

    @patch.dict(os.environ, {"SLURM_NTASKS": "8"}, clear=True)
    def test_resolve_slurm_rank_missing_procid(self):
        """Test returns None when SLURM_PROCID not set."""
        assert resolve_slurm_rank() is None


class TestResolveSLURMWorldSize:
    """Test resolve_slurm_world_size function."""

    @patch.dict(os.environ, {"SLURM_NTASKS": "16"}, clear=True)
    def test_resolve_slurm_world_size(self):
        """Test resolving world size from SLURM_NTASKS."""
        assert resolve_slurm_world_size() == 16

    @patch.dict(os.environ, {}, clear=True)
    def test_resolve_slurm_world_size_not_slurm(self):
        """Test returns None when not in SLURM environment."""
        assert resolve_slurm_world_size() is None


class TestResolveSLURMLocalRank:
    """Test resolve_slurm_local_rank function."""

    @patch.dict(os.environ, {"SLURM_NTASKS": "8", "SLURM_LOCALID": "3"}, clear=True)
    def test_resolve_slurm_local_rank(self):
        """Test resolving local rank from SLURM_LOCALID."""
        assert resolve_slurm_local_rank() == 3

    @patch.dict(os.environ, {}, clear=True)
    def test_resolve_slurm_local_rank_not_slurm(self):
        """Test returns None when not in SLURM environment."""
        assert resolve_slurm_local_rank() is None

    @patch.dict(os.environ, {"SLURM_NTASKS": "8"}, clear=True)
    def test_resolve_slurm_local_rank_missing_localid(self):
        """Test returns None when SLURM_LOCALID not set."""
        assert resolve_slurm_local_rank() is None


class TestParseSLURMNodelist:
    """Test _parse_slurm_nodelist helper function."""

    def test_simple_comma_separated_list(self):
        """Test parsing simple comma-separated nodelist."""
        assert _parse_slurm_nodelist("node001,node002,node003") == "node001"

    def test_simple_single_node(self):
        """Test parsing single node."""
        assert _parse_slurm_nodelist("node042") == "node042"

    def test_bracket_range_notation(self):
        """Test parsing bracket range notation."""
        assert _parse_slurm_nodelist("node[001-004]") == "node001"

    def test_bracket_list_notation(self):
        """Test parsing bracket list notation."""
        assert _parse_slurm_nodelist("node[001,003,005]") == "node001"

    def test_complex_prefix(self):
        """Test parsing with complex prefix."""
        assert _parse_slurm_nodelist("compute-gpu-[10-20]") == "compute-gpu-10"

    def test_with_whitespace(self):
        """Test parsing handles leading/trailing whitespace."""
        assert _parse_slurm_nodelist(" node001 , node002 ") == "node001"

    def test_zero_padded_numbers(self):
        """Test parsing preserves zero-padded numbers."""
        assert _parse_slurm_nodelist("node[001-100]") == "node001"

    def test_mixed_entries_bracket_in_middle(self):
        """First entry is a plain node; a bracketed entry appears later in the list."""
        nodelist = "nodeA,nodeB[001,003],nodeC"
        assert _parse_slurm_nodelist(nodelist) == "nodeA"

    def test_mixed_entries_bracket_first(self):
        """First entry uses bracket expansion, followed by plain nodes."""
        nodelist = "nodeA[001-003],nodeB,nodeC"
        assert _parse_slurm_nodelist(nodelist) == "nodeA001"

    def test_multiple_bracket_groups(self):
        """Multiple bracketed groups in the list — must not merge them."""
        assert _parse_slurm_nodelist("nodeA[001-003],nodeB[005-007]") == "nodeA001"

    def test_mixed_plain_and_bracketed_entries(self):
        """Long list of plain hostnames with a bracketed entry buried in the middle."""
        nodelist = (
            "hostA01,hostA02,hostB01,hostA03,hostA04,hostB02,"
            "hostC[10,12],hostA05,hostB03,hostC04,hostA06,"
            "hostC05,hostA07,hostB04,hostA08"
        )
        assert _parse_slurm_nodelist(nodelist) == "hostA01"


class TestResolveSLURMMasterAddr:
    """Test resolve_slurm_master_addr function."""

    @patch.dict(os.environ, {"SLURM_NTASKS": "4", "SLURM_NODELIST": "node001,node002"}, clear=True)
    def test_simple_comma_list(self):
        """Test parsing simple comma-separated nodelist."""
        assert resolve_slurm_master_addr() == "node001"

    @patch.dict(os.environ, {"SLURM_NTASKS": "4", "SLURM_NODELIST": "node[001-004]"}, clear=True)
    def test_bracket_range(self):
        """Test parsing bracket range nodelist."""
        assert resolve_slurm_master_addr() == "node001"

    @patch.dict(os.environ, {"SLURM_NTASKS": "3", "SLURM_NODELIST": "node[001,003,005]"}, clear=True)
    def test_bracket_list(self):
        """Test parsing bracket list nodelist."""
        assert resolve_slurm_master_addr() == "node001"

    @patch.dict(os.environ, {"SLURM_NTASKS": "4", "SLURM_JOB_NODELIST": "node[010-013]"}, clear=True)
    def test_job_nodelist_fallback(self):
        """Test using SLURM_JOB_NODELIST when SLURM_NODELIST not set."""
        assert resolve_slurm_master_addr() == "node010"

    @patch.dict(os.environ, {}, clear=True)
    def test_not_slurm_environment(self):
        """Test returns None when not in SLURM environment."""
        assert resolve_slurm_master_addr() is None

    @patch.dict(os.environ, {"SLURM_NTASKS": "4"}, clear=True)
    def test_missing_nodelist(self):
        """Test warns and returns localhost when nodelist not set in SLURM environment."""
        with pytest.warns(UserWarning, match="SLURM environment detected.*but SLURM_NODELIST is missing"):
            assert resolve_slurm_master_addr() == "localhost"


class TestResolveSLURMMasterPort:
    """Test resolve_slurm_master_port function."""

    @patch.dict(os.environ, {"SLURM_NTASKS": "8", "SLURM_JOB_ID": "123456"}, clear=True)
    def test_port_from_job_id(self):
        """Test port derived from SLURM_JOB_ID."""
        # Last 4 digits: "3456" + 15000 = 18456
        assert resolve_slurm_master_port() == 18456

    @patch.dict(os.environ, {"SLURM_NTASKS": "8", "SLURM_JOB_ID": "999"}, clear=True)
    def test_port_from_short_job_id(self):
        """Test port derived from short SLURM_JOB_ID."""
        # Last 4 digits of "999": "999" -> int("999") + 15000 = 15999
        assert resolve_slurm_master_port() == 15999

    @patch.dict(os.environ, {"SLURM_NTASKS": "8", "SLURM_JOB_ID": "12345678"}, clear=True)
    def test_port_from_long_job_id(self):
        """Test port derived from long SLURM_JOB_ID."""
        # Last 4 digits: "5678" + 15000 = 20678
        assert resolve_slurm_master_port() == 20678

    @patch.dict(os.environ, {"SLURM_NTASKS": "8"}, clear=True)
    def test_port_without_job_id(self):
        """Test returns None when SLURM_JOB_ID not set."""
        assert resolve_slurm_master_port() is None

    @patch.dict(os.environ, {}, clear=True)
    def test_not_slurm_environment(self):
        """Test returns None when not in SLURM environment."""
        assert resolve_slurm_master_port() is None
