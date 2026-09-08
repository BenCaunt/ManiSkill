// SPDX-License-Identifier: Apache-2.0
// Diagnostic adapter for the additive SAPIEN FromCookedData factory.
#include "vendor/pybind11_conduit_v1.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <sapien/physx/physx_system.h>
#include <sapien/physx/rigid_component.h>
#include <cmath>
#include <cstring>
#include <vector>

namespace py=pybind11;
namespace sp=sapien::physx;
namespace px=physx;

template<class T> T *native(py::handle object) {
  auto *ptr=pybind11_conduit_v1::get_type_pointer_ephemeral<T>(object.ptr());
  if (!ptr) {
    if (PyErr_Occurred()) throw py::error_already_set();
    throw py::type_error("Incompatible SAPIEN native object or C++ ABI");
  }
  return ptr;
}

py::dict describe_mesh(std::shared_ptr<sp::PhysxConvexMesh> const &mesh) {
  auto *raw=mesh->getPxMesh();
  py::dict record;
  py::array_t<float> vertices({py::ssize_t(raw->getNbVertices()),py::ssize_t(3)});
  auto rows=vertices.mutable_unchecked<2>();
  for (px::PxU32 i=0;i<raw->getNbVertices();++i) {
    auto p=raw->getVertices()[i];rows(i,0)=p.x;rows(i,1)=p.y;rows(i,2)=p.z;
  }
  py::array_t<float> planes({py::ssize_t(raw->getNbPolygons()),py::ssize_t(4)});
  auto plane_rows=planes.mutable_unchecked<2>();
  std::vector<uint32_t> indices,offsets{0};
  for (px::PxU32 i=0;i<raw->getNbPolygons();++i) {
    px::PxHullPolygon polygon;
    if (!raw->getPolygonData(i,polygon)) throw std::runtime_error("Polygon readback failed");
    for (int j=0;j<4;++j) plane_rows(i,j)=polygon.mPlane[j];
    for (px::PxU32 j=0;j<polygon.mNbVerts;++j)
      indices.push_back(raw->getIndexBuffer()[polygon.mIndexBase+j]);
    offsets.push_back(indices.size());
  }
  py::array_t<uint32_t> ia(indices.size()),oa(offsets.size());
  std::memcpy(ia.mutable_data(),indices.data(),indices.size()*sizeof(uint32_t));
  std::memcpy(oa.mutable_data(),offsets.data(),offsets.size()*sizeof(uint32_t));
  record["vertices"]=vertices;record["planes"]=planes;
  record["polygon_indices"]=ia;record["polygon_offsets"]=oa;
  record["gpu_compatible"]=raw->isGpuCompatible();
  record["native_reference_count"]=raw->getReferenceCount();
  auto aabb=mesh->getAABB();
  record["mesh_aabb"]=py::make_tuple(py::make_tuple(aabb.lower.x,aabb.lower.y,aabb.lower.z),
                                   py::make_tuple(aabb.upper.x,aabb.upper.y,aabb.upper.z));
  return record;
}

py::list attach(py::object py_system,py::object py_body,py::list blobs,
                py::object py_material,float density) {
  auto *system=native<sp::PhysxSystem>(py_system);
  auto *body=native<sp::PhysxRigidDynamicComponent>(py_body);
  auto material=native<sp::PhysxMaterial>(py_material)->shared_from_this();
  if (!std::isfinite(density) || density<=0) throw py::value_error("Invalid density");
  if (!body->getScene() || body->getPxActor()->getScene()!=system->getPxScene())
    throw py::value_error("Body must belong to the supplied native system");
#ifdef SAPIEN_CUDA
  if (auto gpu=dynamic_cast<sp::PhysxSystemGpu *>(system);gpu && gpu->isInitialized())
    throw py::value_error("Attach cooked geometry before GPU initialization");
#endif
  std::vector<std::shared_ptr<sp::PhysxCollisionShapeConvexMesh>> shapes;
  for (auto blob:blobs) {
    if (!py::isinstance<py::bytes>(blob)) throw py::type_error("Expected cooked bytes");
    auto data=py::cast<std::string>(blob);
    if (data.empty()) throw py::value_error("Empty cooked data");
    auto mesh=sp::PhysxConvexMesh::FromCookedData(std::vector<uint8_t>(data.begin(),data.end()));
    if (system->isGpu() && !mesh->getPxMesh()->isGpuCompatible())
      throw py::value_error("Cooked mesh is not GPU compatible");
    auto shape=std::make_shared<sp::PhysxCollisionShapeConvexMesh>(mesh,sapien::Vec3(1.f),material);
    shape->setDensity(density);
    shapes.push_back(shape);
  }
  // Validate/create every shape before mutating the actor.
  py::list records;
  for (auto const &shape:shapes) {
    body->attachCollision(shape);
    records.append(describe_mesh(shape->getMesh()));
  }
  return records;
}

py::dict inspect(py::object py_shape) {
  auto *shape=native<sp::PhysxCollisionShapeConvexMesh>(py_shape);
  auto mesh=shape->getMesh();
  auto const &geometry=static_cast<px::PxConvexMeshGeometry const &>(shape->getPxShape()->getGeometry());
  auto result=describe_mesh(mesh);
  result["cached_and_native_mesh_same"]=geometry.convexMesh==mesh->getPxMesh();
  auto bound=[](sapien::AABB box) {
    return py::make_tuple(py::make_tuple(box.lower.x,box.lower.y,box.lower.z),
                          py::make_tuple(box.upper.x,box.upper.y,box.upper.z));
  };
  result["local_aabb"]=bound(shape->getLocalAABB());
  result["global_aabb_fast"]=bound(shape->getGlobalAABBFast());
  result["global_aabb_tight"]=bound(shape->computeGlobalAABBTight());
  auto v=mesh->getVertices();auto t=mesh->getTriangles();
  py::array_t<float> va({py::ssize_t(v.rows()),py::ssize_t(3)});
  py::array_t<uint32_t> ta({py::ssize_t(t.rows()),py::ssize_t(3)});
  std::memcpy(va.mutable_data(),v.data(),v.size()*sizeof(float));
  std::memcpy(ta.mutable_data(),t.data(),t.size()*sizeof(uint32_t));
  result["cached_vertices"]=va;result["cached_triangles"]=ta;
  return result;
}

void clone_to(py::object py_system,py::object py_source,py::object py_target) {
  auto *system=native<sp::PhysxSystem>(py_system);
  auto *source=native<sp::PhysxRigidDynamicComponent>(py_source);
  auto *target=native<sp::PhysxRigidDynamicComponent>(py_target);
  if (source==target || source->getPxActor()->getScene()!=system->getPxScene() ||
      target->getPxActor()->getScene()!=system->getPxScene())
    throw py::value_error("Expected separate bodies in the supplied system");
#ifdef SAPIEN_CUDA
  if (auto gpu=dynamic_cast<sp::PhysxSystemGpu *>(system);gpu && gpu->isInitialized())
    throw py::value_error("Clone geometry before GPU initialization");
#endif
  std::vector<std::shared_ptr<sp::PhysxCollisionShape>> clones;
  for (auto const &shape:source->getCollisionShapes()) {
    auto clone=shape->clone();
    // SAPIEN 3.0.3 convex clone omits copyProperties (fixed in supplied source
    // patch). Restore its public properties in this compatibility adapter.
    clone->setDensity(shape->getDensity());
    clone->setCollisionGroups(shape->getCollisionGroups());
    clone->setIsTrigger(shape->isTrigger());
    clone->setIsSceneQuery(shape->isSceneQuery());
    clone->setLocalPose(shape->getLocalPose());
    clone->setContactOffset(shape->getContactOffset());
    clone->setRestOffset(shape->getRestOffset());
    clone->setTorsionalPatchRadius(shape->getTorsionalPatchRadius());
    clone->setMinTorsionalPatchRadius(shape->getMinTorsionalPatchRadius());
    clones.push_back(clone);
  }
  for (auto const &shape:clones) target->attachCollision(shape);
}

PYBIND11_MODULE(sapien303_cooked_bridge,m) {
  m.def("attach",&attach);
  m.def("inspect",&inspect);
  m.def("clone_to",&clone_to);
  m.def("abi",[](){return PYBIND11_PLATFORM_ABI_ID;});
}
