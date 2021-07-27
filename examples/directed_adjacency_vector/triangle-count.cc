//===------------------------------------------------------------*- C++ -*-===//
//
//                                     SHAD
//
//      The Scalable High-performance Algorithms and Data Structure Library
//
//===----------------------------------------------------------------------===//
//
// Copyright 2018 Battelle Memorial Institute
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy
// of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
//===----------------------------------------------------------------------===/

#include <iostream>
#include <iomanip>
#include <random>

#include "shad/core/vector.h"
#include "shad/core/algorithm.h"
#include "shad/core/numeric.h"
#include "shad/core/execution.h"
#include "shad/extensions/graph_library/graph.h"

namespace shad {

struct empty_value {};

int main(int argc, char *argv[]) {

  shad::vector<uint64_t> indptr, indices;
  shad::vector<empty_value> vdata, edata;

  shad::directed_adjacency_vector G(indptr, indices, vdata, edata);

  auto vertices = G.vertices();
  shad::for_each(shad::distributed_parallel_tag{}, vertices.begin(), vertices.end(), [](shad::vertex_iterator v) {
    auto g_oid = v.graph_oid();
    auto G = shad::directed_adjacency_vector::GetPtr(g_oid);
    auto edges = outward_edges(G, v);
    shad::for_each(shad::distributed_parallel_tag{}, edges.begin(), edges.end(), [](shad::edge_iterator e) {
      auto g_oid = e.graph_oid();
      auto G = shad::directed_adjacency_vector::GetPtr(g_oid);
      auto u = vertex(G, e);
      auto edges = outward_edges(G, u);
      shad::for_each(shad::distributed_parallel_tag{}, edges.begin(), edges.end(), [](shad::edge_iterator e) {
          auto g_oid = e.graph_oid();
          auto G = shad::directed_adjacency_vector::GetPtr(g_oid);
          auto w = vertex(G, e);
          auto it = find_outward_edge(G, v, w);
          if (it)
            cnt++;
      });
    });
  });

  return 0;
}

}