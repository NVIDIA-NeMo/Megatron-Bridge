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

import ast
import logging
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


class ParallelismExtractor(ast.NodeVisitor):
    """
    Extracts the parallelism defaults from the default ConfigContainer for a given function name and context.
    """

    _TARGET_ATTRS = {
        "tensor_model_parallel_size": "tensor_model_parallel_size",
        "pipeline_model_parallel_size": "pipeline_model_parallel_size",
        "context_parallel_size": "context_parallel_size",
    }

    def __init__(self, function_name: str, context: Dict[str, Any]) -> None:
        self._function_name = function_name
        self._context = context
        self._scope_depth = 0
        self._context_stack: list[bool] = []
        self.defaults: Dict[str, int] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name != self._function_name:
            return
        self._scope_depth += 1
        self._context_stack.append(True)
        for stmt in node.body:
            self.visit(stmt)
        self._context_stack.pop()
        self._scope_depth -= 1

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._scope_depth == 0 or len(node.targets) != 1 or not self._is_active():
            return

        target_attr = self._match_target(node.targets[0])
        if target_attr is None:
            return

        default_value = self._extract_default(node.value)
        if default_value is not None:
            self.defaults[target_attr] = default_value

    def visit_If(self, node: ast.If) -> None:
        if self._scope_depth == 0 or not self._is_active():
            return

        condition = self._evaluate_condition(node.test)
        if condition is True:
            self._context_stack.append(True)
            for stmt in node.body:
                self.visit(stmt)
            self._context_stack.pop()
        elif condition is False:
            if node.orelse:
                self._context_stack.append(True)
                for stmt in node.orelse:
                    self.visit(stmt)
                self._context_stack.pop()
        else:
            logger.debug(
                "Skipping conditional extraction without evaluable truth value: %s",
                ast.dump(node.test, include_attributes=False),
            )

    def _is_active(self) -> bool:
        return all(self._context_stack) if self._context_stack else True

    def _match_target(self, target: ast.AST) -> Optional[str]:
        if not isinstance(target, ast.Attribute):
            return None

        if target.attr not in self._TARGET_ATTRS:
            return None

        model_attr = target.value
        if not isinstance(model_attr, ast.Attribute) or model_attr.attr != "model":
            return None

        cfg_obj = model_attr.value
        if not isinstance(cfg_obj, ast.Name) or cfg_obj.id != "cfg":
            return None

        return target.attr

    def _extract_default(self, node: ast.AST) -> Optional[int]:
        if isinstance(node, ast.IfExp):
            return self._extract_from_ifexp(node)

        try:
            value = ast.literal_eval(node)
        except Exception:
            logger.debug(
                "Unable to literal-eval node for default parallelism: %s", ast.dump(node, include_attributes=False)
            )
            return None

        if isinstance(value, int):
            return value

        return None

    def _extract_from_ifexp(self, node: ast.IfExp) -> Optional[int]:
        compare = node.test
        if not isinstance(compare, ast.Compare) or len(compare.ops) != 1 or len(compare.comparators) != 1:
            return None

        comparator = compare.comparators[0]
        if not isinstance(comparator, (ast.Constant, ast.NameConstant)) or comparator.value is not None:
            return None

        op = compare.ops[0]
        try:
            if isinstance(op, (ast.Is, ast.Eq)):
                candidate = ast.literal_eval(node.body)
            elif isinstance(op, (ast.IsNot, ast.NotEq)):
                candidate = ast.literal_eval(node.orelse)
            else:
                return None
        except Exception:
            logger.debug(
                "Failed to evaluate conditional parallelism default: %s", ast.dump(node, include_attributes=False)
            )
            return None

        if isinstance(candidate, int):
            return candidate

        return None

    def _evaluate_condition(self, node: ast.AST) -> Optional[bool]:
        expr = ast.Expression(body=node)
        try:
            compiled = compile(ast.fix_missing_locations(expr), "<parallelism_condition>", "eval")
            result = eval(compiled, {"__builtins__": {}}, self._context)
        except Exception:
            logger.debug(
                "Unable to evaluate condition for parallelism extraction: %s",
                ast.dump(node, include_attributes=False),
            )
            return None

        if isinstance(result, bool):
            return result

        return None
