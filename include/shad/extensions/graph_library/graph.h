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
//===----------------------------------------------------------------------===//

#ifndef INCLUDE_SHAD_EXTENSIONS_GRAPH_LIBRARY_GRAPH_H_
#define INCLUDE_SHAD_EXTENSIONS_GRAPH_LIBRARY_GRAPH_H_

#include <algorithm>
#include <functional>
#include <tuple>
#include <utility>
#include <vector>
#include <memory>
#include <type_traits>
#include <tuple>

#include "shad/data_structures/abstract_data_structure.h"
#include "shad/core/vector.h"
#include "shad/runtime/runtime.h"

namespace shad {

struct empty_value {};

namespace impl {

template <  typename VV                            = empty_value,
            typename EV                            = empty_value,
            typename GV                            = empty_value,
            typename KeyT                          = std::uint32_t,
            template <typename V> class VContainer = shad::vector,
            template <typename E> class EContainer = shad::vector>
class directed_adjacency_vector {
  class BaseVertexRef;
  class VertexRef;

  class BaseEdgeRef;
  class EdgeRef;

  class vertex_iterator;

  class edge_iterator;

public:

  using size_type = std::size_t;
  
  using graph_value_type = GV;

  using vertex_key_type = KeyT;
  using vertex_value_type = VV;

  struct vertex_type {
    vertex_value_type val;
    vertex_key_type id;

    vertex_value_type& value() {
      return val;
    }

    const vertex_value_type& value() const {
      return val;
    }
  }

  using edge_key_type = size_type;
  using edge_value_type = EV;

  struct edge_type {
    edge_value_type val;
    vertex_key_type sid;
    vertex_key_type did;
    edge_key_type eid;

    edge_type(vertex_key_type sid, edge_key_type eid) : val(data[eid]), sid(sid), did(indices[eid]), eid(eid) {}
 
    edge_value_type& value() {
      return val;
    }

    const edge_value_type& value() const {
      return val;
    }

    void read() {
      val = data[eid];
    }

    void write() const {
      data[eid] = val;
    }
  }

  using indptrObjectID_t = VContainer<size_type>::ObjectID;
  using indicesObjectID_t = EContainer<vertex_key_type>::ObjectID;
  using vertex_dataObjectID_t = VContainer<vertex_value_type>::ObjectID;
  using edge_dataObjectID_t = EContainer<edge_value_type>::ObjectID;

 public:
    directed_adjacency_vector()                                 = default;
    directed_adjacency_vector(indptrObjectID_t indptr_oid, indicesObjectID_t indices_oid, vertex_dataObjectID_t vertex_data_oid, dataObjectID_t edge_data_oid) : 
      indptr(VContainer<size_type>::GetPtr(indptr_oid)),
      indices(EContainer<vertex_key_type>::GetPtr(indices_oid)),
      vertex_data(vertex_data_oid != vertex_dataObjectID_t::kNullID ? VContainer<vertex_value_type>::GetPtr(vertex_data_oid) : nullptr),
      edge_data(edge_data_oid != edge_dataObjectID_t::kNullID ? EContainer<edge_value_type>::GetPtr(edge_data_oid) : nullptr) {}

 private:
  VContainer<size_type> indptr;
  EContainer<vertex_key_type> indices;
  // graph_value_type graph_data;
  VContainer<vertex_value_type> vertex_data;
  EContainer<edge_value_type> edge_data;
};

template <  typename VV                            = empty_value,
            typename EV                            = empty_value,
            typename GV                            = empty_value,
            typename KeyT                          = std::uint32_t,
            template <typename V> class VContainer = shad::vector,
            template <typename E> class EContainer = shad::vector>
class alignas(64) directed_adjacency_vector<VV, EV, GV, KeyT, VContainer, EContainer>::vertex_iterator {
  using graph_type = directed_adjacency_vector<VV, EV, GV, KeyT, VContainer, EContainer>;
 public:
  using reference = typename graph_type::VertexRef;
  using difference_type = std::make_signed_t<graph_type::vertex_key_type>;
  using value_type = typename graph_type::vertex_type;
  using reference = typename graph_type::VertexRef;
  using prefix_type = typename VContainer<graph_type::size_type>::difference_type;

  /// @brief Constructor.
  vertex_iterator(rt::Locality &&l, difference_type offset, ObjectID oid, prefix_type const *p)
      : locality_(l), offset_(offset), oid_(oid), p_(p) {}

  /// @brief Default constructor.
  vertex_iterator()
      : vertex_iterator(rt::Locality(0), -1, ObjectID::kNullID, nullptr) {}

  /// @brief Copy constructor.
  vertex_iterator(const vertex_iterator &O)
      : locality_(O.locality_),
        offset_(O.offset_),
        oid_(O.oid_),
        p_(O.p_) {}

  /// @brief Move constructor.
  vertex_iterator(const vertex_iterator &&O) noexcept
      : locality_(std::move(O.locality_)),
        offset_(std::move(O.offset_)),
        oid_(std::move(O.oid_)),
        p_(std::move(O.p_)) {}

  /// @brief Copy assignment operator.
  vertex_iterator &operator=(const vertex_iterator &O) {
    locality_ = O.locality_;
    offset_ = O.offset_;
    oid_ = O.oid_;
    p_ = O.p_;

    return *this;
  }

  /// @brief Move assignment operator.
  vertex_iterator &operator=(vertex_iterator &&O) {
    locality_ = std::move(O.locality_);
    offset_ = std::move(O.offset_);
    oid_ = std::move(O.oid_);
    p_ = std::move(O.p_);

    return *this;
  }

  bool operator==(const vertex_iterator &O) const {
    return locality_ == O.locality_ && oid_ == O.oid_ && offset_ == O.offset_;
  }

  bool operator!=(const vertex_iterator &O) const { return !(*this == O); }

  reference operator*() {// @todo
    return reference(locality_, offset_, oid_);
  }

  vertex_iterator &operator++() {
    std::uint32_t l = locality_;
    const auto g_offset = p_[l] + offset_ + 1;
    if (g_offset < p_[l + 1])
      ++offset_;
    else {
      const auto num_l = rt::numLocalities();
      while (l < num_l && g_offset >= p_[l + 1])
        l++;
      if (l == num_l) {
        locality_ = rt::Locality(num_l - 1);
        offset_ = p_[num_l] - p_[num_l - 1];
      }
      else {
        locality_ = rt::Locality(l);
        offset_ = 0;
      }
    }
    return *this;
  }

  vertex_iterator operator++(int) {
    vertex_iterator tmp(*this);
    operator++();
    return tmp;
  }

  vertex_iterator &operator--() {
    if (offset_ > 0)
      --offset_;
    else {
      std::uint32_t l = locality_;
      const difference_type g_offset = p_[l] - 1;
      if (g_offset < 0) {
        locality_ = rt::Locality(0);
        offset_ = -1;
      }
      else {
        while(g_offset < p_[l - 1])
          l--;
        locality_ = rt::Locality(l - 1);
        offset_ = p_[l] - p_[l - 1] - 1;
      }
    }
    
    return *this;
  }

  vertex_iterator operator--(int) {
    vector_iterator tmp(*this);
    operator--();
    return tmp;
  }

  vertex_iterator &operator+=(difference_type n) {
    const std::uint32_t l = locality_;
    const auto g_offset = p_[l] + offset_ + n;
    if (p_[l] <= g_offset && g_offset < p_[l + 1])
      offset_ += n;
    else {
      const auto num_l = rt::numLocalities();
      const auto l = shad::vector<graph_type::size_type>::lowerbound_index(p_, p_ + num_l + 1, g_offset);
      if (l < 0) {
        locality_ = rt::Locality(0);
        offset_ = -1;
      }
      else if (l >= num_l) {
        locality_ = rt::Locality(num_l - 1);
        offset_ = p_[num_l] - p_[num_l - 1];
      }
      else {
        locality_ = rt::Locality(l);
        offset_ = g_offset - p_[l];
      }
    }

    return *this;
  }

  vertex_iterator &operator-=(difference_type n) {
    return operator+=(-n);
  }

  vertex_iterator operator+(difference_type n) {
    vertex_iterator tmp(*this);
    return tmp.operator+=(n);
  }

  vertex_iterator operator-(difference_type n) {
    vertex_iterator tmp(*this);
    return tmp.operator-=(n);
  }

  difference_type operator-(const vertex_iterator &O) const {
    if (oid_ != O.oid_) return std::numeric_limits<difference_type>::min();

    return get_global_id() - O.get_global_id();
  }

  bool operator<(const vertex_iterator &O) const {
    if (oid_ != O.oid_ || locality_ > O.locality_) return false;
    return locality_ < O.locality_ || offset_ < O.offset_;
  }

  bool operator>(const vertex_iterator &O) const {
    if (oid_ != O.oid_ || locality_ < O.locality_) return false;
    return locality_ > O.locality_ || offset_ > O.offset_;
  }

  bool operator<=(const vertex_iterator &O) const { return !(*this > O); }

  bool operator>=(const vertex_iterator &O) const { return !(*this < O); }

  friend std::ostream &operator<<(std::ostream &stream,
                                  const vertex_iterator i) {
    stream << i.locality_ << " " << i.offset_;
    return stream;
  }

 protected:
  constexpr difference_type get_global_id() const {
    return p_[to_int(locality_)] + offset_;
  }

 private:
  rt::Locality locality_;
  ObjectID oid_;
  difference_type offset_;
  prefix_type const *p_;
};

}  // namespace impl




template <  typename VV                                        = empty_value,
            typename EV                                        = empty_value,
            typename GV                                        = empty_value,
            typename KeyT                                      = std::uint32_t,
            template <typename V, typename A> class VContainer = shad::vector,
            template <typename E, typename A> class EContainer = shad::vector>
struct graph_traits<impl::directed_adjacency_vector<VV, EV, GV, KeyT, VContainer, EContainer>> {

  using graph_type = impl::directed_adjacency_vector<VV, EV, GV, KeyT, VContainer, EContainer>;
  using graph_value_type = typename graph_type::graph_value_type;

  using vertex_type = typename graph_type::vertex_type;
  using vertex_key_type = typename graph_type::vertex_key_type;
  using vertex_value_type = typename graph_type::vertex_value_type;

  using edge_type = typename graph_type::edge_type;
  using edge_key_type = typename graph_type::edge_key_type;
  using edge_value_type = typename graph_type::edge_value_type;

  using vertex_range = typename graph_type::vertex_range;
  using const_vertex_range = typename graph_type::const_vertex_range;

  using edge_range = typename graph_type::edge_range;
  using const_edge_range = typename graph_type::const_edge_range;

  using vertex_outward_edge_range = typename graph_type::vertex_outward_edge_range;
  using const_vertex_outward_edge_range = typename graph_type::const_vertex_outward_edge_range;

  using vertex_outward_vertex_range = typename graph_type::vertex_outward_vertex_range;
  using const_vertex_outward_vertex_range = typename graph_type::vertex_outward_vertex_range;

  using vertex_edge_range = vertex_outward_edge_range;
  using const_vertex_edge_range = const_vertex_edge_range;

  using vertex_vertex_range = vertex_outward_vertex_range;
  using const_vertex_vertex_range = const_vertex_outward_vertex_range;

};


template <  typename VV                            = empty_value,
            typename EV                            = empty_value,
            typename GV                            = empty_value,
            typename KeyT                          = std::uint32_t,
            template <typename V> class VContainer = shad::vector,
            template <typename E> class EContainer = shad::vector>
class directed_adjacency_vector {
  using graph_t = impl::directed_adjacency_vector<VV, EV, GV, KeyT, VContainer, EContainer>;

 public:
  using ObjectID = graph_t::ObjectID;

  /// @defgroup Types
  /// @{
  using size_type = typename graph_t::size_type;
  using graph_value_type = typename graph_t::graph_value_type;
  using vertex_key_type = typename graph_t::vertex_key_type;
  using vertex_value_type = typename graph_t::vertex_value_type;
  using vertex_type = typename graph_t::vertex_type;
  using edge_key_type = typename graph_t::edge_key_type;
  using edge_value_type = typename graph_t::edge_value_type;
  using edge_type = typename graph_t::edge_type;

 public:
  explicit directed_adjacency_vector(graph_t::SharedPtr ptr) : ptr(ptr) {}

  /// @brief Constructor.
  explicit directed_adjacency_vector( VContainer<size_type> indptr, EContainer<vertex_key_type> indices, VContainer<vertex_value_type> vertex_data, EContainer<edge_value_type> edge_data) { 
    ptr = graph_t::Create(indptr.get_oid(), indices.get_oid(), vertex_data.get_oid(), edge_data.get_oid());
  }

  /// @brief Destructor.
  ~directed_adjacency_vector() { graph_t::Destroy(impl()->GetGlobalID()); }

  /// @brief The copy assignment operator.
  ///
  /// @param O The right-hand side of the operator.
  /// @return A reference to the left-hand side.
  directed_adjacency_vector &operator=(const directed_adjacency_vector &O) {
    impl()->operator=(*O.ptr);
    return *this;
  }

 private:
  graph_t::SharedPtr ptr = nullptr;

  const graph_t *impl() const { return ptr.get(); }

  graph_t *impl() { return ptr.get(); }

  friend bool operator==(const directed_adjacency_vector, const directed_adjacency_vector &RHS) {
    return *LHS.ptr == *RHS.ptr;
  }

  friend bool operator<(const directed_adjacency_vector, const directed_adjacency_vector &RHS) {
    return operator<(*LHS.ptr, *RHS.ptr);
  }

  friend bool operator>(const directed_adjacency_vector, const directed_adjacency_vector &RHS) {
    return operator>(*LHS.ptr, *RHS.ptr);
  }

}


struct A {
  explicit A(int x) {}
};

struct B : public A {
  using A::A;
};

template <class G>
struct vertex_iterator : public G::vertex_iterator {
  using G::vertex_iterator::vertex_iterator;
};

template <class G>
struct const_vertex_iterator : public G::vertex_iterator {
  using G::vertex_iterator::vertex_iterator;
};

template <directed G>
constexpr auto outward_degree(const G& g, const_vertex_iterator_t<G> u) noexcept
      //-> vertex_outward_edge_size_t<G> 
{
  return G.
}
      
template <directed G>
constexpr auto outward_edges(G& g, vertex_iterator_t<G> u)
      -> vertex_outward_edge_range_t<G>;
template <directed G>
constexpr auto outward_edges(const G&, const_vertex_iterator_t<G> u) -> const_vertex_outward_edge_range_t<G>;
template <directed G>
constexpr auto find_outward_edge(G& g, vertex_iterator_t<G> u, vertex_iterator_t<G> v)
      -> vertex_outward_edge_iterator_t<G>;
template <directed G>
constexpr auto find_outward_edge(const G&, const_vertex_iterator_t<G> u,
                                 const_vertex_iterator_t<G> v)
      -> const_vertex_outward_edge_iterator_t<G>;
template <directed G>
constexpr auto find_outward_edge(G& g, vertex_key_t<G> ukey, vertex_key_t<G> vkey)
      -> vertex_outward_edge_iterator_t<G>;
template <directed G>
constexpr auto find_outward_edge(const G&, vertex_key_t<G> ukey, vertex_key_t<G> vkey)
-> const_vertex_outward_edge_iterator_t<G>;


// Vertex-Outward-Vertex Functions
template <directed G>
constexpr auto outward_vertices(G& g, vertex_iterator_t<G> u)
      -> vertex_outward_vertex_range_t<G>;
template <directed G>
constexpr auto outward_vertices(const G&, const_vertex_iterator_t<G> u) -> const_vertex_outward_vertex_range_t<G>;
template <directed G>
constexpr auto
find_outward_vertex(G& g, vertex_iterator_t<G> u, vertex_iterator_t<G> v)
      -> vertex_outward_vertex_iterator_t<G>;
template <directed G>
constexpr auto find_outward_vertex(const G&, const_vertex_iterator_t<G> u,
                                   const_vertex_iterator_t<G> v)
      -> const_vertex_outward_vertex_iterator_t<G>;
template <directed G>
constexpr auto find_outward_vertex(G& g, vertex_key_t<G> ukey, vertex_key_t<G> vkey)
      -> vertex_outward_vertex_iterator_t<G>;
template <directed G>
constexpr auto
find_outward_vertex(const G&, vertex_key_t<G> ukey, vertex_key_t<G> vkey)
      -> const_vertex_outward_vertex_iterator_t<G>;

// Modifying Functions
template <directed G>
constexpr void clear_outward_edges(G& g, vertex_iterator_t<G> u);

}  // namespace shad

#endif  // INCLUDE_SHAD_EXTENSIONS_GRAPH_LIBRARY_GRAPH_H_
