// Read-only convex cooking diagnostic using public SAPIEN/PhysX interfaces.
#include "../vendor/pybind11_conduit_v1.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <sapien/physx/physx_system.h>
#include <sapien/physx/physx_engine.h>
#include <memory>
#include <cmath>
#include <vector>

namespace py=pybind11;
namespace px=physx;
using Rows=py::array_t<float,py::array::c_style>;

py::dict cook_polyhedron(py::object object, Rows vertices, Rows planes,
                         py::array_t<uint32_t,py::array::c_style> indices,
                         py::array_t<uint32_t,py::array::c_style> offsets, bool gpu) {
  auto *system=pybind11_conduit_v1::get_type_pointer_ephemeral<sapien::physx::PhysxSystem>(object.ptr());
  if (!system) {
    if (PyErr_Occurred()) throw py::error_already_set();
    throw py::type_error("Incompatible SAPIEN system or C++ ABI");
  }
  if (vertices.ndim()!=2 || vertices.shape(1)!=3 || vertices.shape(0)<4 || vertices.shape(0)>255 ||
      planes.ndim()!=2 || planes.shape(1)!=4 || planes.shape(0)<4 || planes.shape(0)>255 ||
      indices.ndim()!=1 || offsets.ndim()!=1 || offsets.size()!=planes.shape(0)+1 ||
      indices.size()>65535 || offsets.data()[0]!=0 || offsets.data()[offsets.size()-1]!=indices.size())
    throw py::value_error("Invalid polyhedron descriptor layout");
  for (py::ssize_t i=0;i<vertices.size();++i)
    if (!std::isfinite(vertices.data()[i])) throw py::value_error("Nonfinite convex vertex");
  auto equations=planes.unchecked<2>();
  std::vector<px::PxHullPolygon> polygons(planes.shape(0));
  for (py::ssize_t i=0;i<planes.shape(0);++i) {
    auto begin=offsets.data()[i],end=offsets.data()[i+1];
    if (end<begin || end>indices.size() || end-begin<3 || end-begin>255)
      throw py::value_error("Invalid polygon offsets");
    for (auto j=begin;j<end;++j)
      if (indices.data()[j]>=vertices.shape(0)) throw py::value_error("Out-of-range convex index");
    float norm=0.f;
    for (int j=0;j<4;++j) {
      if (!std::isfinite(equations(i,j))) throw py::value_error("Nonfinite convex plane");
      polygons[i].mPlane[j]=equations(i,j);
      if (j<3) norm+=equations(i,j)*equations(i,j);
    }
    if (std::abs(norm-1.f)>1e-3f) throw py::value_error("Nonunit convex plane");
    polygons[i].mNbVerts=end-begin;polygons[i].mIndexBase=begin;
  }
  auto *physics=system->getEngine()->getPxPhysics();
  px::PxConvexMeshDesc desc;
  desc.points.count=vertices.shape(0);desc.points.stride=3*sizeof(float);desc.points.data=vertices.data();
  desc.polygons.count=polygons.size();desc.polygons.stride=sizeof(px::PxHullPolygon);desc.polygons.data=polygons.data();
  desc.indices.count=indices.size();desc.indices.stride=sizeof(uint32_t);desc.indices.data=indices.data();
  // No eCOMPUTE_CONVEX: preserve the supplied convex polyhedron.
  px::PxCookingParams params(physics->getTolerancesScale());params.buildGPUData=gpu;
  px::PxDefaultMemoryOutputStream output;px::PxConvexMeshCookingResult::Enum status;
  if (!PxCookConvexMesh(params,desc,output,&status)) throw std::runtime_error("Explicit native convex cooking failed");
  px::PxDefaultMemoryInputData input(output.getData(),output.getSize());
  auto release=[](px::PxConvexMesh *p){if(p)p->release();};
  std::unique_ptr<px::PxConvexMesh,decltype(release)> mesh(physics->createConvexMesh(input),release);
  if (!mesh) throw std::runtime_error("Explicit cooked mesh deserialization failed");
  Rows result({py::ssize_t(mesh->getNbVertices()),py::ssize_t(3)});
  auto rows=result.mutable_unchecked<2>();auto *points=mesh->getVertices();
  for (px::PxU32 i=0;i<mesh->getNbVertices();++i) {
    rows(i,0)=points[i].x;rows(i,1)=points[i].y;rows(i,2)=points[i].z;
  }
  py::dict record;record["vertices"]=result;record["status"]=int(status);
  record["polygon_count"]=mesh->getNbPolygons();record["gpu_compatible"]=mesh->isGpuCompatible();
  record["build_gpu_data"]=params.buildGPUData;
  Rows actual_planes({py::ssize_t(mesh->getNbPolygons()),py::ssize_t(4)});
  auto plane_rows=actual_planes.mutable_unchecked<2>();
  std::vector<uint32_t> polygon_indices,polygon_offsets{0};
  auto *native_indices=mesh->getIndexBuffer();
  for (px::PxU32 i=0;i<mesh->getNbPolygons();++i) {
    px::PxHullPolygon polygon;
    if (!mesh->getPolygonData(i,polygon)) throw std::runtime_error("Native polygon readback failed");
    for (int j=0;j<4;++j) plane_rows(i,j)=polygon.mPlane[j];
    for (px::PxU32 j=0;j<polygon.mNbVerts;++j) polygon_indices.push_back(native_indices[polygon.mIndexBase+j]);
    polygon_offsets.push_back(polygon_indices.size());
  }
  py::array_t<uint32_t> index_array(polygon_indices.size()),offset_array(polygon_offsets.size());
  std::memcpy(index_array.mutable_data(),polygon_indices.data(),polygon_indices.size()*sizeof(uint32_t));
  std::memcpy(offset_array.mutable_data(),polygon_offsets.data(),polygon_offsets.size()*sizeof(uint32_t));
  record["planes"]=actual_planes;record["polygon_indices"]=index_array;record["polygon_offsets"]=offset_array;
  record["cooked"]=py::bytes(reinterpret_cast<const char *>(output.getData()),output.getSize());
  return record;
}

py::dict cook(py::object object, Rows vertices, float plane_tolerance, bool gpu) {
  auto *system=pybind11_conduit_v1::get_type_pointer_ephemeral<sapien::physx::PhysxSystem>(object.ptr());
  if (!system) {
    if (PyErr_Occurred()) throw py::error_already_set();
    throw py::type_error("Incompatible SAPIEN system or C++ ABI");
  }
  if (vertices.ndim()!=2 || vertices.shape(1)!=3 || vertices.shape(0)<4 ||
      !std::isfinite(plane_tolerance) || plane_tolerance<0) throw py::value_error("Invalid convex cooking input");
  for (py::ssize_t i=0;i<vertices.size();++i)
    if (!std::isfinite(vertices.data()[i])) throw py::value_error("Nonfinite convex vertex");
  auto *physics=system->getEngine()->getPxPhysics();
  px::PxConvexMeshDesc desc;
  desc.points.count=vertices.shape(0);desc.points.stride=3*sizeof(float);desc.points.data=vertices.data();
  desc.flags=px::PxConvexFlag::eCOMPUTE_CONVEX;desc.vertexLimit=255;
  px::PxCookingParams params(physics->getTolerancesScale());
  params.planeTolerance=plane_tolerance;params.buildGPUData=gpu;
  px::PxDefaultMemoryOutputStream output;
  px::PxConvexMeshCookingResult::Enum status;
  if (!PxCookConvexMesh(params,desc,output,&status)) throw std::runtime_error("Native convex cooking failed");
  px::PxDefaultMemoryInputData input(output.getData(),output.getSize());
  auto release=[](px::PxConvexMesh *p){if(p)p->release();};
  std::unique_ptr<px::PxConvexMesh,decltype(release)> mesh(physics->createConvexMesh(input),release);
  if (!mesh) throw std::runtime_error("Native cooked mesh deserialization failed");
  Rows result({py::ssize_t(mesh->getNbVertices()),py::ssize_t(3)});
  auto rows=result.mutable_unchecked<2>();auto *points=mesh->getVertices();
  for (px::PxU32 i=0;i<mesh->getNbVertices();++i) {
    rows(i,0)=points[i].x;rows(i,1)=points[i].y;rows(i,2)=points[i].z;
  }
  py::dict record;
  record["vertices"]=result;record["status"]=int(status);
  record["polygon_count"]=mesh->getNbPolygons();record["plane_tolerance"]=params.planeTolerance;
  record["gpu"]=params.buildGPUData;record["cooked_bytes"]=output.getSize();
  return record;
}

PYBIND11_MODULE(sapien303_polyhedron_bridge,m) {
  m.def("cook",&cook,py::arg("system"),py::arg("vertices").noconvert(),py::arg("plane_tolerance"),py::arg("gpu"));
  m.def("cook_polyhedron",&cook_polyhedron,py::arg("system"),py::arg("vertices").noconvert(),
        py::arg("planes").noconvert(),py::arg("indices").noconvert(),py::arg("offsets").noconvert(),py::arg("gpu"));
}
