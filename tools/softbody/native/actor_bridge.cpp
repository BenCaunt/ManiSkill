// Native SAPIEN 3.0.3 selected-actor diagnostic. No private object layouts.
#include "vendor/pybind11_conduit_v1.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <sapien/physx/physx_system.h>
#include <sapien/physx/rigid_component.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstring>
#include <set>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
namespace sp = sapien::physx;
namespace px = physx;

template<class T> T *native(py::handle obj) {
  auto *ptr = pybind11_conduit_v1::get_type_pointer_ephemeral<T>(obj.ptr());
  if (!ptr) {
    if (PyErr_Occurred()) throw py::error_already_set();
    throw py::type_error("Incompatible SAPIEN native object or C++ ABI");
  }
  return ptr;
}

void cuda_check(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
struct DeviceScope {
  int previous{-1};
  explicit DeviceScope(int device) {
    cuda_check(cudaGetDevice(&previous));
    cuda_check(cudaSetDevice(device));
    cuda_check(cudaDeviceSynchronize());
  }
  ~DeviceScope() { if (previous >= 0) cudaSetDevice(previous); }
};
template<class T> struct DeviceArray {
  T *ptr{nullptr};
  explicit DeviceArray(size_t count) {
    if (count) cuda_check(cudaMalloc(reinterpret_cast<void **>(&ptr), count * sizeof(T)));
  }
  ~DeviceArray() { if (ptr) cudaFree(ptr); }
  DeviceArray(DeviceArray const &) = delete;
  DeviceArray &operator=(DeviceArray const &) = delete;
};

struct Selection {
  sp::PhysxSystemGpu *system;
  std::vector<sp::PhysxRigidDynamicComponent *> actors;
  std::vector<px::PxGpuActorPair> pairs;
  std::vector<sapien::Vec3> offsets;
  Selection(py::handle py_system, py::list py_actors) {
    system = native<sp::PhysxSystemGpu>(py_system);
    system->checkGpuInitialized();
    std::set<px::PxRigidDynamic *> seen;
    for (auto item : py_actors) {
      auto *body = native<sp::PhysxRigidDynamicComponent>(item);
      auto *actor = body->getPxActor();
      if (actor->getScene() != system->getPxScene())
        throw py::value_error("Actor does not belong to the selected native system");
      if (!seen.insert(actor).second) throw py::value_error("Duplicate native actor");
      auto node = actor->getInternalIslandNodeIndex();
      if (!node.isValid() || node.isArticulation())
        throw py::value_error("Expected an initialized rigid dynamic actor");
      px::PxGpuActorPair pair;
      std::memset(&pair, 0, sizeof(pair));
      pair.srcIndex = static_cast<px::PxU32>(actors.size());
      pair.nodeIndex = node;
      pairs.push_back(pair);
      actors.push_back(body);
      offsets.push_back(system->getSceneOffset(body->getScene()));
    }
  }
};

using Rows = py::array_t<float, py::array::c_style>;
Rows read_actors(py::object system, py::list actors) {
  Selection s(system, actors);
  py::ssize_t n = static_cast<py::ssize_t>(s.actors.size());
  Rows result({n, py::ssize_t(13)});
  if (!n) return result;
  DeviceScope device(s.system->getDevice()->cudaId);
  DeviceArray<px::PxGpuBodyData> data(n);
  DeviceArray<px::PxGpuActorPair> pairs(n);
  cuda_check(cudaMemcpy(pairs.ptr, s.pairs.data(), n*sizeof(px::PxGpuActorPair), cudaMemcpyHostToDevice));
  s.system->getPxScene()->copyBodyData(data.ptr, pairs.ptr, n);
  cuda_check(cudaDeviceSynchronize());
  std::vector<px::PxGpuBodyData> host(n);
  cuda_check(cudaMemcpy(host.data(), data.ptr, n*sizeof(px::PxGpuBodyData), cudaMemcpyDeviceToHost));
  auto out = result.mutable_unchecked<2>();
  for (py::ssize_t i=0;i<n;++i) {
    auto const &d=host[i]; auto o=s.offsets[i];
    float row[13]={d.pos.x-o.x,d.pos.y-o.y,d.pos.z-o.z,
      d.quat.w,d.quat.x,d.quat.y,d.quat.z,
      d.linVel.x,d.linVel.y,d.linVel.z,d.angVel.x,d.angVel.y,d.angVel.z};
    for (int j=0;j<13;++j) out(i,j)=row[j];
  }
  return result;
}

void apply_actors(py::object system, py::list actors, Rows rows) {
  Selection s(system, actors);
  size_t n=s.actors.size();
  if (rows.ndim()!=2 || rows.shape(0)!=static_cast<py::ssize_t>(n) || rows.shape(1)!=13)
    throw py::value_error("Expected a contiguous float32 actor state with shape (N,13)");
  if (!n) return;
  auto src=rows.unchecked<2>();
  std::vector<px::PxGpuBodyData> host(n);
  for (size_t i=0;i<n;++i) {
    for (int j=0;j<13;++j) if (!std::isfinite(src(i,j))) throw py::value_error("Nonfinite actor state");
    auto &d=host[i]; auto o=s.offsets[i];
    d.quat=px::PxQuat(src(i,4),src(i,5),src(i,6),src(i,3));
    if (std::abs(d.quat.magnitudeSquared()-1.f)>1e-3f) throw py::value_error("Nonunit actor quaternion");
    d.pos=px::PxVec4(src(i,0)+o.x,src(i,1)+o.y,src(i,2)+o.z,0.f);
    d.linVel=px::PxVec4(src(i,7),src(i,8),src(i,9),0.f);
    d.angVel=px::PxVec4(src(i,10),src(i,11),src(i,12),0.f);
  }
  DeviceScope device(s.system->getDevice()->cudaId);
  DeviceArray<px::PxGpuBodyData> data(n);
  DeviceArray<px::PxGpuActorPair> pairs(n);
  cuda_check(cudaMemcpy(data.ptr,host.data(),n*sizeof(px::PxGpuBodyData),cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pairs.ptr,s.pairs.data(),n*sizeof(px::PxGpuActorPair),cudaMemcpyHostToDevice));
  // Pair srcIndex indexes this compact data buffer, not SAPIEN's full buffer.
  s.system->getPxScene()->applyActorData(data.ptr,pairs.ptr,px::PxActorCacheFlag::eACTOR_DATA,n);
  cuda_check(cudaDeviceSynchronize());
}

// Joint buffers use global articulation-index rows. SAPIEN 3.0.3's indexed
// joint methods submit the prefix of its internal index buffer by mistake.
// Keep the full native data buffer and submit only the actual selection.
void apply_articulation_data(py::object py_system, std::vector<int> const &selected,
                             bool targets) {
  auto *system = native<sp::PhysxSystemGpu>(py_system);
  system->checkGpuInitialized();
  int count = system->getArticulationCount();
  std::set<int> seen;
  std::vector<px::PxU32> host_indices;
  for (int index : selected) {
    if (index < 0 || index >= count || !seen.insert(index).second)
      throw py::value_error("Expected unique in-range initialized articulation indices");
    host_indices.push_back(static_cast<px::PxU32>(index));
  }
  if (host_indices.empty()) return;
  DeviceScope device(system->getDevice()->cudaId);
  DeviceArray<px::PxU32> indices(host_indices.size());
  cuda_check(cudaMemcpy(indices.ptr, host_indices.data(), host_indices.size()*sizeof(px::PxU32),
                        cudaMemcpyHostToDevice));
  auto apply = [&](void *data, px::PxArticulationGpuDataType::Enum field) {
    system->getPxScene()->applyArticulationData(data, indices.ptr, field,
                                               static_cast<px::PxU32>(host_indices.size()));
  };
  if (targets) {
    apply(system->gpuGetArticulationQTargetPosCudaHandle().ptr, px::PxArticulationGpuDataType::eJOINT_TARGET_POSITION);
    apply(system->gpuGetArticulationQTargetVelCudaHandle().ptr, px::PxArticulationGpuDataType::eJOINT_TARGET_VELOCITY);
  } else {
    apply(system->gpuGetArticulationQposCudaHandle().ptr, px::PxArticulationGpuDataType::eJOINT_POSITION);
    apply(system->gpuGetArticulationQvelCudaHandle().ptr, px::PxArticulationGpuDataType::eJOINT_VELOCITY);
    apply(system->gpuGetArticulationQfCudaHandle().ptr, px::PxArticulationGpuDataType::eJOINT_FORCE);
  }
  cuda_check(cudaDeviceSynchronize());
}

py::dict describe(py::object system, py::list actors) {
  Selection s(system,actors);
  py::list records;
  for (size_t i=0;i<s.actors.size();++i) {
    auto *a=s.actors[i]; auto cm=a->getPxActor()->getCMassLocalPose();
    py::dict record;
    record["gpu_index"]=a->getGpuIndex();
    record["node_index"]=s.pairs[i].nodeIndex.getInd();
    record["src_index"]=s.pairs[i].srcIndex;
    record["offset"]=py::make_tuple(s.offsets[i].x,s.offsets[i].y,s.offsets[i].z);
    record["cmass_local_pose"]=py::make_tuple(cm.p.x,cm.p.y,cm.p.z,cm.q.w,cm.q.x,cm.q.y,cm.q.z);
    records.append(record);
  }
  py::dict result;
  result["abi"]=PYBIND11_PLATFORM_ABI_ID;
  result["body_data_size"]=sizeof(px::PxGpuBodyData);
  result["pair_size"]=sizeof(px::PxGpuActorPair);
  result["actors"]=records;
  return result;
}

PYBIND11_MODULE(sapien303_actor_bridge,m) {
  m.def("read_actors", &read_actors);
  m.def("apply_actors", &apply_actors,py::arg("system"),py::arg("actors"),py::arg("states").noconvert());
  m.def("apply_articulation_data", &apply_articulation_data, py::arg("system"), py::arg("indices"), py::arg("targets"));
  m.def("describe", &describe);
  m.def("abi", [](){ return PYBIND11_PLATFORM_ABI_ID; });
}
