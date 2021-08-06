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
#include <atomic>
#include <fstream>

#include "shad/core/vector.h"
#include "shad/core/algorithm.h"
#include "shad/core/numeric.h"
#include "shad/core/execution.h"
#include "shad/extensions/graph_library/graph.h"

namespace shad {

template <typename F>
struct lambda_wrapper {
 lambda_wrapper() = default;
 
 lambda_wrapper(F const& f) { std::memcpy(buffer, &f, sizeof(F)); }
 
 template <typename... Ts>
 auto operator()(Ts&&... ts) {
 return (*reinterpret_cast<F*>(buffer))(std::forward<Ts>(ts)...);
 }
 
 template <typename... Ts>
 auto operator()(Ts&&... ts) const {
 return (*reinterpret_cast<F const*>(buffer))(std::forward<Ts>(ts)...);
 }
 
 std::uint8_t buffer[sizeof(F)];
};

auto get_matrix_type(std::istream &in) {
    char s[4];
    in.read(s, sizeof s);
    assert(std::isdigit(s[0]));
    assert(std::isdigit(s[2]));
    s[1] = s[3] = '\0';
    return std::make_pair(std::stoi(s), std::stoi(s + 2));
}

template <typename Ordinal>
auto read_binary(std::istream &in) {
    std::vector<Ordinal> indptr;
    std::vector<Ordinal> indices;

    const auto [v_dtype, e_dtype] = get_matrix_type(in);
    assert(v_dtype == sizeof(Ordinal) && e_dtype == sizeof(Ordinal));
    Ordinal N = v_dtype + e_dtype;
    Ordinal M;
    in.read(reinterpret_cast<char *>(&N), sizeof(Ordinal));
    in.read(reinterpret_cast<char *>(&M), sizeof(Ordinal));
    indptr.resize(N + 1);
    in.read(reinterpret_cast<char*>(indptr.data()), indptr.size() * sizeof(Ordinal));
    indices.resize(indptr.back());
    in.read(reinterpret_cast<char *>(indices.data()), indices.size() * sizeof(Ordinal));

    return std::make_tuple(indptr, indices);
}


int main(int argc, char *argv[]) {
  if (argc <= 1)
    return 0;

  std::ifstream in(argv[1]);

  auto [indptr_data, indices_data] = read_binary<uint32_t>(in);

  shad::vector<uint64_t> indptr(indptr_data.size()), indices(indices_data.size());
  for(int i = 0; i < indptr_data.size(); i++)
    indptr[i] = indptr_data[i];

  for(int i = 0; i < indices_data.size(); i++)
    indices[i] = indices_data[i];

  std::cerr << indptr_data.size() << " " << indices_data.size() << std::endl;
  shad::vector<uint32_t> vdata(indptr_data.size() - 1), edata(indices_data.size());

  shad::directed_adjacency_vector<uint32_t, uint32_t, uint32_t, uint64_t> G(indptr, indices, vdata, edata);
  using graph_t = decltype(G);

  static std::atomic<std::size_t> cnt = 0;
  auto vertices = G.vertices();
  shad::for_each(shad::distributed_parallel_tag{}, vertices.begin(), vertices.end(), [](shad::vertex_iterator_t<graph_t> v) {
    auto g_oid = v.get_oid();
    auto G = graph_t::GetPtr(g_oid);
    auto edges = outward_edges(G, v);
    shad::for_each(shad::distributed_parallel_tag{}, edges.begin(), edges.end(), lambda_wrapper([v](shad::edge_iterator_t<graph_t> e) {
      auto g_oid = e.get_oid();
      auto G = graph_t::GetPtr(g_oid);
      auto u = vertex(G, e);
      auto edges = outward_edges(G, u);
      shad::for_each(shad::distributed_parallel_tag{}, edges.begin(), edges.end(), lambda_wrapper([v](shad::edge_iterator_t<graph_t> e) {
          auto g_oid = e.get_oid();
          auto G = graph_t::GetPtr(g_oid);
          auto w = vertex(G, e);
          auto it = find_outward_edge(G, v, w);
          if (it)
            cnt++;
      }));
    }));
  });

  return 0;
}

}