# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Exercise coverage routing and revision binding with PyYAML and Node.js."""

import json
import logging
import subprocess
from pathlib import Path

import yaml


workflow = yaml.safe_load(Path(".github/workflows/cicd-main.yml").read_text())
jobs = workflow["jobs"]
scope = next(step for step in jobs["configure"]["steps"] if step.get("id") == "coverage-scope")
status = next(step for step in jobs["Coverage_Fake"]["steps"] if step["name"] == "Generate fake coverage report")
uploader = next(step for step in jobs["Coverage"]["steps"] if step["name"] == "Upload coverage reports to Codecov")
payload = {
    "scopeScript": scope["with"]["script"],
    "statusScript": status["with"]["script"],
    "fakeCondition": jobs["Coverage_Fake"]["if"],
    "uploadCondition": jobs["Coverage"]["if"],
    "configuredHead": jobs["configure"]["outputs"]["pr_head_sha"],
    "uploadHead": uploader["with"]["override_commit"],
    "flags": jobs["Coverage"]["strategy"]["matrix"]["flag"],
}
harness = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const data = JSON.parse(fs.readFileSync(0, 'utf8'));
const AsyncFunction = Object.getPrototypeOf(async function() {}).constructor;
const classify = new AsyncFunction('github', 'context', 'core', data.scopeScript);
const context = {sha: 'head', repo: {owner: 'NVIDIA-NeMo', repo: 'Megatron-Bridge'}};
const allowed = ['.main.commit', '.dev.commit', '3rdparty/Megatron-LM', 'uv.lock'];
const evaluateRevision = (expression, configuredHead, lookupHead, workflowSha) => {
  const steps = {'get-pr-info': {outputs: {'pr-info': JSON.stringify(
    lookupHead ? {head: {sha: lookupHead}} : {})}}};
  const needs = {configure: {outputs: {pr_head_sha: configuredHead}}};
  const github = {sha: workflowSha};
  const body = expression.slice(3, -2)
    .replaceAll('steps.get-pr-info', 'steps["get-pr-info"]')
    .replaceAll('.outputs.pr-info', '.outputs["pr-info"]')
    .replaceAll('.head.sha', '?.head?.sha');
  return new Function('needs', 'steps', 'github', 'fromJSON', `return (${body});`)(
    needs, steps, github, JSON.parse);
};
const cases = [
  ['MCore main bump', ['.main.commit', '3rdparty/Megatron-LM', 'uv.lock'], true],
  ['MCore dev bump', ['.dev.commit', '3rdparty/Megatron-LM', 'uv.lock'], true],
  ['lock only', ['uv.lock'], true],
  ['all allowed paths', allowed, true],
  ['mixed source change', ['uv.lock', 'src/megatron/bridge/training/train.py'], false],
  ['dependency declarations', ['pyproject.toml', 'uv.lock'], false],
  ['similar prefix', ['uv.lock.extra'], false],
  ['submodule child', ['3rdparty/Megatron-LM/file.py'], false],
  ['empty diff', [], false],
  ['rename from source', [{filename:'uv.lock', previous_filename:'src/source.py'}], false],
];
(async () => {
  for (const [configuredLookup, laterLookup, workflowSha] of [
    ['head-a', 'head-b', 'head-a'],
    ['head-a', 'head-a', 'head-a'],
    [null, null, 'main-sha'],
    [null, null, 'merge-group-sha'],
  ]) {
    const configuredHead = evaluateRevision(data.configuredHead, null, configuredLookup, workflowSha);
    for (const flag of data.flags) {
      assert.equal(evaluateRevision(data.uploadHead, configuredHead, laterLookup, workflowSha),
        configuredLookup || workflowSha, `wrong upload revision for ${flag}`);
    }
  }
  for (const [name, paths, expected] of cases) {
    const files = paths.map(path => typeof path === 'string' ? {filename:path} : path);
    process.env.PR_INFO = JSON.stringify({number:6810, base:{sha:'base'}, head:{sha:'head'}, changed_files:files.length});
    let result;
    const github = {rest:{repos:{compareCommitsWithBasehead:async params => {
      assert.equal(params.basehead, 'base...head');
      return {data:{files}};
    }}}};
    await classify(github, context, {setOutput:(_, value) => result = value});
    assert.equal(result, String(expected), name);
  }
  for (const pr of [{}, {number:6810, head:{sha:'stale'}},
    {number:6810,base:{sha:'base'},head:{sha:'head'},changed_files:2}]) {
    process.env.PR_INFO = JSON.stringify(pr);
    let result;
    await classify({rest:{repos:{compareCommitsWithBasehead:async()=>({data:{files:[{filename:'uv.lock'}]}})}}}, context,
      {setOutput:(_,value)=>result=value});
    assert.equal(result, 'false');
  }
  process.env.PR_INFO = JSON.stringify({number:6810,base:{sha:'base'},head:{sha:'head'},changed_files:1});
  await assert.rejects(classify({rest:{repos:{compareCommitsWithBasehead:async()=>{throw Error('API failed')}}}},context,
    {setOutput:()=>assert.fail('API failure must not emit exemption')}), /API failed/);
  const evaluate = (expression, dependencyOnly, result, cancelled=false, docsOnly=false) => {
    const needs = {configure:{outputs:{docs_only:String(docsOnly),perf_scripts_only:'false',dependency_only:String(dependencyOnly)}},
      'pre-flight':{outputs:{is_ci_workload:'false',is_deployment_workflow:'false'}},Nemo_CICD_Test:{result}};
    expression = expression.replaceAll('needs.pre-flight', 'needs["pre-flight"]');
    return new Function('needs','github','always','cancelled',`return (${expression});`)(needs,
      {event:{inputs:{mcore_ref:''}}},()=>true,()=>cancelled);
  };
  assert.equal(evaluate(data.fakeCondition,true,'success'),true);
  for (const result of ['failure','cancelled','skipped']) {
    assert.equal(evaluate(data.fakeCondition,true,result),false);
    assert.equal(evaluate(data.uploadCondition,true,result),false);
  }
  assert.equal(evaluate(data.uploadCondition,true,'success'),false);
  assert.equal(evaluate(data.fakeCondition,false,'success'),false);
  assert.equal(evaluate(data.uploadCondition,false,'success'),true);
  assert.equal(evaluate(data.fakeCondition,true,'success',true),false);
  process.env.PR_HEAD_SHA = 'new-untested-head';
  for (const docsOnly of [false, true]) {
    assert.equal(evaluate(data.fakeCondition,!docsOnly,'success',false,docsOnly),true);
    let posted;
    await new AsyncFunction('github','context',data.statusScript)(
      {rest:{repos:{createCommitStatus:async value=>posted=value}}}, context);
    assert.equal(posted.sha,context.sha,'exemption must not move to a newer, untested PR head');
    assert.equal(posted.context,'codecov/patch');
    assert.equal(posted.state,'success');
  }
  console.log('PASS: revision binding for both flags, 14 classifier scenarios, coverage routing, workflow-head exemptions');
})().catch(error=>{console.error(error);process.exitCode=1});
"""
subprocess.run(["node", "-e", harness], input=json.dumps(payload), text=True, check=True)
assert "override_pr" in uploader["with"]
assert ".number" in uploader["with"]["override_pr"]
assert uploader["with"]["override_commit"] == "${{ needs.configure.outputs.pr_head_sha }}"
assert "base_sha" not in uploader["with"]
assert "dependency_only" not in jobs["configure"]["steps"][-1]["run"]
logging.basicConfig(level=logging.INFO)
logging.getLogger(__name__).info("PASS: explicit uploader metadata; test selection unchanged")
