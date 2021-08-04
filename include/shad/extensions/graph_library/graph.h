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
            typename KeyT                          = std::uint32_t>
class directed_adjacency_vector : public AbstractDataStructure<directed_adjacency_vector<VV, EV, GV, KeyT>> {
  // class BaseVertexRef;
  // class VertexRef;

  // class BaseEdgeRef;
  // class EdgeRef;

 public:

  template <typename T>
  using VContainer = shad::vector<T>;
  template <typename T>
  using EContainer = shad::vector<T>;

  using size_type = std::size_t;

  /// The type for the global identifier.
  using ObjectID = typename AbstractDataStructure<directed_adjacency_vector<VV, EV, GV, KeyT>>::ObjectID;
  using difference_type = typename VContainer<size_type>::difference_type;

  class vertex_iterator;
  class vertex_range {
    vertex_iterator b, e;
  
   public:

    vertex_range(vertex_iterator b, vertex_iterator e) : b(b), e(e) {}

    auto begin() {
      return b;
    }

    auto end() {
      return e;
    }
  };

  class edge_iterator;
  class edge_range {
    edge_iterator b, e;
  
   public:

    edge_range(edge_iterator b, edge_iterator e) : b(b), e(e) {}

    auto begin() {
      return b;
    }

    auto end() {
      return e;
    }
  };
  
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
  };

  using edge_key_type = size_type;
  using edge_value_type = EV;

  struct edge_type {
    edge_value_type val;
    vertex_key_type sid;
    vertex_key_type did;
    edge_key_type eid;

    edge_type(vertex_key_type sid, edge_key_type eid) : val(edge_data[eid]), sid(sid), did(indices[eid]), eid(eid) {}
 
    edge_value_type& value() {
      return val;
    }

    const edge_value_type& value() const {
      return val;
    }

    void read() {
      val = edge_data[eid];
    }

    void write() const {
      edge_data[eid] = val;
    }
  };

  using indptrObjectID_t = typename VContainer<size_type>::ObjectID;
  using indicesObjectID_t = typename EContainer<vertex_key_type>::ObjectID;
  using vertex_dataObjectID_t = typename VContainer<vertex_value_type>::ObjectID;
  using edge_dataObjectID_t = typename EContainer<edge_value_type>::ObjectID;

 public:
  /// @brief DataStructure identifier getter.
  ///
  /// Returns the global object identifier associated to a DataStructure
  /// instance.
  ///
  /// @warning It must be implemented in the inheriting DataStructure.
  ObjectID GetGlobalID() const { return oid_; }

  directed_adjacency_vector(ObjectID oid) : oid_(oid) {}
  directed_adjacency_vector(ObjectID oid, indptrObjectID_t indptr_oid, indicesObjectID_t indices_oid, vertex_dataObjectID_t vertex_data_oid, edge_dataObjectID_t edge_data_oid) : 
    oid_(oid),
    indptr(VContainer<size_type>::GetPtr(indptr_oid)),
    indices(EContainer<vertex_key_type>::GetPtr(indices_oid)),
    vertex_data(vertex_data_oid != vertex_dataObjectID_t::kNullID ? VContainer<vertex_value_type>::GetPtr(vertex_data_oid) : nullptr),
    edge_data(edge_data_oid != edge_dataObjectID_t::kNullID ? EContainer<edge_value_type>::GetPtr(edge_data_oid) : nullptr) {}
    
  vertex_key_type degree(const vertex_key_type i) const {
    return indices[i + 1] - indices[i];
  }

  auto begin() const {
    return vertex_iterator(0, oid_);
  }

  auto end() const {
    return vertex_iterator(indptr.size() - 1, oid_);
  }

  auto begin(const vertex_key_type i) const {
    return edge_iterator(indptr[i], oid_);
  }

  auto end(const vertex_key_type i) const {
    return begin(i + 1);
  }

  vertex_key_type vertex_key(edge_iterator et) const {
    edge_key_type ei = et;
    return indices[ei];
  }

  auto vertex(edge_iterator et) const {
    edge_key_type ei = et;
    return begin() + indices[ei];
  }

  bool find_outward_edge(const vertex_iterator u, const vertex_iterator v) const {
    auto begin = indices.begin() + indptr[(vertex_key_type)u];
    auto end = indices.begin() + indptr[(vertex_key_type)u + 1];
    return end != shad::find(shad::distributed_parallel_tag{}, begin, end, (vertex_key_type)v);
  }

 private:
  ObjectID oid_;
  VContainer<size_type> indptr;
  EContainer<vertex_key_type> indices;
  // graph_value_type graph_data;
  VContainer<vertex_value_type> vertex_data;
  EContainer<edge_value_type> edge_data;
};

template <  typename VV,
            typename EV,
            typename GV,
            typename KeyT>
class alignas(64) directed_adjacency_vector<VV, EV, GV, KeyT>::vertex_iterator {
  using graph_type = directed_adjacency_vector<VV, EV, GV, KeyT>;
 public:
  // using reference = typename graph_type::VertexRef;
  using difference_type = std::make_signed_t<graph_type::vertex_key_type>;
  using value_type = typename graph_type::vertex_type;
  using prefix_type = typename VContainer<graph_type::size_type>::difference_type;

  using iterator_category = std::random_access_iterator_tag;

  using local_iterator_type = vertex_iterator;

  using distribution_range = std::vector<std::pair<rt::Locality, size_t>>;

  /// @brief Constructor.
  vertex_iterator(difference_type offset, ObjectID oid)
      : offset_(offset), oid_(oid) {}

  /// @brief Default constructor.
  vertex_iterator()
      : vertex_iterator(-1, ObjectID::kNullID) {}

  /// @brief Copy constructor.
  vertex_iterator(const vertex_iterator &O)
      : offset_(O.offset_),
        oid_(O.oid_) {}

  /// @brief Move constructor.
  vertex_iterator(const vertex_iterator &&O) noexcept
      : offset_(std::move(O.offset_)),
        oid_(std::move(O.oid_)) {}

  explicit operator graph_type::vertex_key_type() const {
    return offset_;
  }

  /// @brief Copy assignment operator.
  vertex_iterator &operator=(const vertex_iterator &O) {
    offset_ = O.offset_;
    oid_ = O.oid_;

    return *this;
  }

  /// @brief Move assignment operator.
  vertex_iterator &operator=(vertex_iterator &&O) {
    offset_ = std::move(O.offset_);
    oid_ = std::move(O.oid_);

    return *this;
  }

  bool operator==(const vertex_iterator &O) const {
    return oid_ == O.oid_ && offset_ == O.offset_;
  }

  bool operator!=(const vertex_iterator &O) const { return !(*this == O); }

  auto operator*() {// @todo
    return this;
  }

  vertex_iterator &operator++() {
    offset_++;

    return *this;
  }

  vertex_iterator operator++(int) {
    vertex_iterator tmp(*this);
    operator++();
    return tmp;
  }

  vertex_iterator &operator--() {
    --offset_;
    
    return *this;
  }

  vertex_iterator operator--(int) {
    vertex_iterator tmp(*this);
    operator--();
    return tmp;
  }

  vertex_iterator &operator+=(difference_type n) {
    offset_ += n;

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
    if (oid_ != O.oid_) return false;
    return offset_ < O.offset_;
  }

  bool operator>(const vertex_iterator &O) const {
    if (oid_ != O.oid_) return false;
    return offset_ > O.offset_;
  }

  bool operator<=(const vertex_iterator &O) const { return !(*this > O); }

  bool operator>=(const vertex_iterator &O) const { return !(*this < O); }

  class local_iterator_range {
   public:
    local_iterator_range(local_iterator_type B, local_iterator_type E)
        : begin_(B), end_(E) {}
    local_iterator_type begin() { return begin_; }
    local_iterator_type end() { return end_; }

   private:
    local_iterator_type begin_;
    local_iterator_type end_;
  };

  static local_iterator_range local_range(vertex_iterator &B,
                                          vertex_iterator &E) {
    auto begin{B.get_local_chunk()};
    
    if (B.oid_ != E.oid_ || rt::thisLocality() < B.get_locality() || rt::thisLocality() > E.get_locality())
      return local_iterator_range(begin, begin);

    if (B.get_locality() == rt::thisLocality())
      begin += B.offset_;
    
    auto p = B.get_p();

    const std::uint32_t l = rt::thisLocality();
    const auto chunk = p[l + 1] - p[l];

    auto end{B.get_local_chunk() + chunk};
    if (E.get_locality() == rt::thisLocality())
      end = B.get_local_chunk() + E.offset_;
    return local_iterator_range(begin, end);
  }

  static distribution_range
      distribution(vertex_iterator &begin, vertex_iterator &end) {
    distribution_range result;

    // First block:
    const std::uint32_t begin_l = begin.get_locality();
    auto p = begin.get_p();
    const auto start_block_size = p[begin_l + 1] - p[begin_l];
    if (begin.get_locality() == end.get_locality())
      start_block_size = end.offset_;
    result.push_back(std::make_pair(begin.get_locality(),
                                    start_block_size - begin.offset_));

    // Middle blocks:
    for (auto locality = begin.get_locality() + 1;
         locality < end.get_locality(); ++locality) {
      const std::uint32_t mid_l = locality;
      const auto inner_block_size = p[mid_l + 1] - p[mid_l];
      result.push_back(std::make_pair(locality, inner_block_size));
    }

    // Last block:
    if (end.offset_ != 0 && begin.get_locality() != end.get_locality())
      result.push_back(std::make_pair(end.get_locality(), end.offset_));

    return result;
  }

  static rt::localities_range localities(vertex_iterator &B, vertex_iterator &E) {
    return rt::localities_range(
        B.get_locality(), rt::Locality(static_cast<uint32_t>(E.get_locality()) + (E.offset_ != 0 ? 1 : 0)));
  }

  static vertex_iterator iterator_from_local(vertex_iterator &B,
                                            vertex_iterator &E,
                                            local_iterator_type itr) {
    if (rt::thisLocality() < B.get_locality() || rt::thisLocality() > E.get_locality())
      return E;

    return vertex_iterator(rt::thisLocality(),
                          std::distance(B.get_local_chunk(), itr), B.oid_,
                          B.ptrs_, B.p_);
  }

  friend std::ostream &operator<<(std::ostream &stream,
                                  const vertex_iterator i) {
    stream << i.offset_;
    return stream;
  }

  ObjectID get_oid() const {
    return oid_;
  }

 protected:
  constexpr difference_type get_global_id() const {
    return offset_;
  }

 private:

  auto get_p() const {
    auto g = graph_type::GetPtr(oid_);
    return g.indptr.get_p();
  }

  rt::Locality get_locality() const {
    auto p = get_p();
    return shad::lowerbound_index(p, p + rt::numLocalities() + 1, offset_);
  }

  vertex_iterator get_local_chunk() const {
    auto p = get_p();
    int l = rt::thisLocality();
    return vertex_iterator(p[l], oid_);
  }

  ObjectID oid_;
  difference_type offset_;
};

template <  typename VV,
            typename EV,
            typename GV,
            typename KeyT>
class alignas(64) directed_adjacency_vector<VV, EV, GV, KeyT>::edge_iterator {
  using graph_type = directed_adjacency_vector<VV, EV, GV, KeyT>;
 public:
  // using reference = typename graph_type::VertexRef;
  using difference_type = std::make_signed_t<graph_type::edge_key_type>;
  using value_type = typename graph_type::vertex_type;
  using prefix_type = typename VContainer<graph_type::size_type>::difference_type;

  using iterator_category = std::random_access_iterator_tag;

  using local_iterator_type = edge_iterator;

  using distribution_range = std::vector<std::pair<rt::Locality, size_t>>;

  /// @brief Constructor.
  edge_iterator(difference_type offset, ObjectID oid)
      : offset_(offset), oid_(oid) {}

  /// @brief Default constructor.
  edge_iterator()
      : edge_iterator(-1, ObjectID::kNullID) {}

  /// @brief Copy constructor.
  edge_iterator(const edge_iterator &O)
      : offset_(O.offset_),
        oid_(O.oid_) {}

  /// @brief Move constructor.
  edge_iterator(const edge_iterator &&O) noexcept
      : offset_(std::move(O.offset_)),
        oid_(std::move(O.oid_)) {}

  explicit operator graph_type::edge_key_type() const {
    return get_global_id();
  }

  /// @brief Copy assignment operator.
  edge_iterator &operator=(const edge_iterator &O) {
    offset_ = O.offset_;
    oid_ = O.oid_;

    return *this;
  }

  /// @brief Move assignment operator.
  edge_iterator &operator=(edge_iterator &&O) {
    offset_ = std::move(O.offset_);
    oid_ = std::move(O.oid_);

    return *this;
  }

  bool operator==(const edge_iterator &O) const {
    return oid_ == O.oid_ && offset_ == O.offset_;
  }

  bool operator!=(const edge_iterator &O) const { return !(*this == O); }

  edge_iterator operator*() {// @todo
    return *this;
  }

  edge_iterator &operator++() {
    offset_++;

    return *this;
  }

  edge_iterator operator++(int) {
    edge_iterator tmp(*this);
    operator++();
    return tmp;
  }

  edge_iterator &operator--() {
    --offset_;

    return *this;
  }

  edge_iterator operator--(int) {
    vertex_iterator tmp(*this);
    operator--();
    return tmp;
  }

  edge_iterator &operator+=(difference_type n) {
    offset_++;

    return *this;
  }

  edge_iterator &operator-=(difference_type n) {
    return operator+=(-n);
  }

  edge_iterator operator+(difference_type n) {
    edge_iterator tmp(*this);
    return tmp.operator+=(n);
  }

  edge_iterator operator-(difference_type n) {
    edge_iterator tmp(*this);
    return tmp.operator-=(n);
  }

  difference_type operator-(const edge_iterator &O) const {
    if (oid_ != O.oid_) return std::numeric_limits<difference_type>::min();

    return get_global_id() - O.get_global_id();
  }

  ObjectID get_oid() const {
    return oid_;
  }

  bool operator<(const edge_iterator &O) const {
    if (oid_ != O.oid_) return false;
    return offset_ < O.offset_;
  }

  bool operator>(const edge_iterator &O) const {
    if (oid_ != O.oid_) return false;
    return offset_ > O.offset_;
  }

  bool operator<=(const edge_iterator &O) const { return !(*this > O); }

  bool operator>=(const edge_iterator &O) const { return !(*this < O); }

  class local_iterator_range {
   public:
    local_iterator_range(local_iterator_type B, local_iterator_type E)
        : begin_(B), end_(E) {}
    local_iterator_type begin() { return begin_; }
    local_iterator_type end() { return end_; }

   private:
    local_iterator_type begin_;
    local_iterator_type end_;
  };

  static local_iterator_range local_range(edge_iterator &B,
                                          edge_iterator &E) {
    auto begin{B.get_local_chunk()};
    
    if (B.oid_ != E.oid_ || rt::thisLocality() < B.get_locality() || rt::thisLocality() > E.get_locality())
      return local_iterator_range(begin, begin);

    if (B.get_locality() == rt::thisLocality())
      begin += B.offset_;

    const std::uint32_t l = rt::thisLocality();
    const auto chunk = B.p_[l + 1] - B.p_[l];

    auto end{B.get_local_chunk() + chunk};
    if (E.get_locality() == rt::thisLocality())
      end = B.get_local_chunk() + E.offset_;
    return local_iterator_range(begin, end);
  }

  static distribution_range
      distribution(edge_iterator &begin, edge_iterator &end) {
    distribution_range result;

    // First block:
    const std::uint32_t begin_l = begin.get_locality();
    const auto start_block_size = begin.p_[begin_l + 1] - begin.p_[begin_l];
    if (begin.get_locality() == end.get_locality())
      start_block_size = end.offset_;
    result.push_back(std::make_pair(begin.get_locality(),
                                    start_block_size - begin.offset_));

    // Middle blocks:
    for (auto locality = begin.get_locality() + 1;
         locality < end.get_locality(); ++locality) {
      const std::uint32_t mid_l = locality;
      const auto inner_block_size = begin.p_[mid_l + 1] - begin.p_[mid_l];
      result.push_back(std::make_pair(locality, inner_block_size));
    }

    // Last block:
    if (end.offset_ != 0 && begin.get_locality() != end.get_locality())
      result.push_back(std::make_pair(end.get_locality(), end.offset_));

    return result;
  }

  static rt::localities_range localities(edge_iterator &B, edge_iterator &E) {
    return rt::localities_range(
        B.get_locality(), rt::Locality(static_cast<uint32_t>(E.get_locality()) + (E.offset_ != 0 ? 1 : 0)));
  }

  static edge_iterator iterator_from_local(edge_iterator &B,
                                            edge_iterator &E,
                                            local_iterator_type itr) {
    if (rt::thisLocality() < B.get_locality() || rt::thisLocality() > E.get_locality())
      return E;

    return edge_iterator(rt::thisLocality(),
                          std::distance(B.get_local_chunk(), itr), B.oid_,
                          B.ptrs_, B.p_);
  }
  
  friend std::ostream &operator<<(std::ostream &stream,
                                  const edge_iterator i) {
    stream << i.offset_;
    return stream;
  }

 protected:
  constexpr difference_type get_global_id() const {
    return offset_;
  }

 private:

  rt::Locality get_locality() const {
    auto g = graph_type::GetPtr(oid_);
    auto p = g.indptr.get_p();
    return shad::lowerbound_index(p, p + rt::numLocalities() + 1, offset_);
  }

  vertex_iterator get_local_chunk() const {
    auto g = graph_type::GetPtr(oid_);
    auto p = g.indptr.get_p();
    int l = rt::thisLocality();
    return vertex_iterator(p[l], oid_);
  }

  ObjectID oid_;
  difference_type offset_;
};

}  // namespace impl

template <typename G>
struct graph_traits {

  using graph_type = G;
  using graph_value_type = typename graph_type::graph_value_type;

  using vertex_type = typename graph_type::vertex_type;
  using vertex_key_type = typename graph_type::vertex_key_type;
  using vertex_value_type = typename graph_type::vertex_value_type;

  using edge_type = typename graph_type::edge_type;
  using edge_key_type = typename graph_type::edge_key_type;
  using edge_value_type = typename graph_type::edge_value_type;

  using vertex_range = typename graph_type::vertex_range;
  using const_vertex_range = typename graph_type::vertex_range;

  using edge_range = typename graph_type::edge_range;
  using const_edge_range = typename graph_type::edge_range;

  // using vertex_outward_edge_range = typename graph_type::vertex_outward_edge_range;
  // using const_vertex_outward_edge_range = typename graph_type::const_vertex_outward_edge_range;

  // using vertex_outward_vertex_range = typename graph_type::vertex_outward_vertex_range;
  // using const_vertex_outward_vertex_range = typename graph_type::vertex_outward_vertex_range;

  // using vertex_edge_range = vertex_outward_edge_range;
  // using const_vertex_edge_range = const_vertex_edge_range;

  // using vertex_vertex_range = vertex_outward_vertex_range;
  // using const_vertex_vertex_range = const_vertex_outward_vertex_range;
};

struct A {
  explicit A(int x) {}
};

struct B : public A {
  using A::A;
};

template <class G>
using vertex_iterator_t = typename G::vertex_iterator;

template <class G>
using const_vertex_iterator_t = typename G::vertex_iterator;

template <class G>
using edge_iterator_t = typename G::edge_iterator;

template <class G>
using const_edge_iterator_t = typename G::edge_iterator;

template <  typename VV                            = empty_value,
            typename EV                            = empty_value,
            typename GV                            = empty_value,
            typename KeyT                          = std::uint32_t>
class directed_adjacency_vector {
  using graph_t = impl::directed_adjacency_vector<VV, EV, GV, KeyT>;
  template <typename T>
  using VContainer = shad::vector<T>;
  template <typename T>
  using EContainer = shad::vector<T>;

 public:
  using ObjectID = typename graph_t::ObjectID;
  using SharedPtr = typename graph_t::SharedPtr;

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
  using edge_range = typename graph_t::edge_range;
  using vertex_range = typename graph_t::vertex_range;
  using edge_iterator = typename graph_t::edge_iterator;
  using vertex_iterator = typename graph_t::vertex_iterator;

 public:
  explicit directed_adjacency_vector(SharedPtr ptr) : ptr(ptr) {}

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

  static directed_adjacency_vector GetPtr(const ObjectID oid) {
    return directed_adjacency_vector(graph_t::GetPtr(oid));
  }

  vertex_key_type degree(const vertex_key_type i) const {
    return ptr->degree(i);
  }

  edge_range outward_edges(const vertex_key_type i) const {
    return edge_range(ptr->begin(i), ptr->end(i));
  }

  bool find_outward_edge(const vertex_iterator u, const vertex_iterator v) const {
    return ptr->find_outward_edge(u, v);
  }

  vertex_range vertices() const {
    return vertex_range(ptr->begin(), ptr->end());
  }

  vertex_key_type vertex_key(edge_iterator et) const {
    return ptr->vertex_key(et);
  }

  vertex_iterator vertex(edge_iterator et) const {
    return vertex_iterator(ptr->vertex_key(et), ptr->GetGlobalID());
  }

 private:
  SharedPtr ptr = nullptr;

  const graph_t *impl() const { return ptr.get(); }

  graph_t *impl() { return ptr.get(); }

  friend bool operator==(const directed_adjacency_vector &LHS, const directed_adjacency_vector &RHS) {
    return *LHS.ptr == *RHS.ptr;
  }

  friend bool operator<(const directed_adjacency_vector &LHS, const directed_adjacency_vector &RHS) {
    return operator<(*LHS.ptr, *RHS.ptr);
  }

  friend bool operator>(const directed_adjacency_vector &LHS, const directed_adjacency_vector &RHS) {
    return operator>(*LHS.ptr, *RHS.ptr);
  }
};

template <typename G, typename I>
  // requires (edge_iterator<G,I> || adjacency_iterator<G,I>)
constexpr auto vertex(G& g, I uv) // -> vertex_iterator_t<G>;
{
  return g.vertex(uv);
}

template <typename G>
constexpr auto outward_degree(const G& g, const_vertex_iterator_t<G> u) noexcept
      // -> vertex_outward_edge_size_t<G> 
{
  typename graph_traits<G>::vertex_key_type i = u;
  return g.degree(i);
}
      
template <typename G>
constexpr auto outward_edges(G& g, vertex_iterator_t<G> u)
      // -> vertex_outward_edge_range_t<G>;
{
  typename graph_traits<G>::vertex_key_type i = u;
  return g.outward_edges(i);
}
template <typename G>
constexpr auto find_outward_edge(G& g, vertex_iterator_t<G> u, vertex_iterator_t<G> v)
//       -> vertex_outward_edge_iterator_t<G>;
{
  return g.find_outward_edge(u, v);
}
// template <typename G>
// constexpr auto find_outward_edge(const G&, const_vertex_iterator_t<G> u,
//                                  const_vertex_iterator_t<G> v)
//       -> const_vertex_outward_edge_iterator_t<G>;
// template <typename G>
// constexpr auto find_outward_edge(G& g, vertex_key_t<G> ukey, vertex_key_t<G> vkey)
//       -> vertex_outward_edge_iterator_t<G>;
// template <typename G>
// constexpr auto find_outward_edge(const G&, vertex_key_t<G> ukey, vertex_key_t<G> vkey)
// -> const_vertex_outward_edge_iterator_t<G>;


// // Vertex-Outward-Vertex Functions
// template <typename G>
// constexpr auto outward_vertices(G& g, vertex_iterator_t<G> u)
//       -> vertex_outward_vertex_range_t<G>;
// template <typename G>
// constexpr auto outward_vertices(const G&, const_vertex_iterator_t<G> u) -> const_vertex_outward_vertex_range_t<G>;
// template <typename G>
// constexpr auto
// find_outward_vertex(G& g, vertex_iterator_t<G> u, vertex_iterator_t<G> v)
//       -> vertex_outward_vertex_iterator_t<G>;
// template <typename G>
// constexpr auto find_outward_vertex(const G&, const_vertex_iterator_t<G> u,
//                                    const_vertex_iterator_t<G> v)
//       -> const_vertex_outward_vertex_iterator_t<G>;
// template <typename G>
// constexpr auto find_outward_vertex(G& g, vertex_key_t<G> ukey, vertex_key_t<G> vkey)
//       -> vertex_outward_vertex_iterator_t<G>;
// template <typename G>
// constexpr auto
// find_outward_vertex(const G&, vertex_key_t<G> ukey, vertex_key_t<G> vkey)
//       -> const_vertex_outward_vertex_iterator_t<G>;

}  // namespace shad

#endif  // INCLUDE_SHAD_EXTENSIONS_GRAPH_LIBRARY_GRAPH_H_
