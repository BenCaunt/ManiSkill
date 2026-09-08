// SPDX-License-Identifier: Apache-2.0
// Additive factory for trusted, version-matched PhysX cooked convex meshes.
#include "sapien/physx/mesh.h"
#include "sapien/physx/physx_engine.h"
#include <algorithm>
#include <cstring>
#include <limits>

namespace {
class CookedInput final : public physx::PxInputData {
public:
  explicit CookedInput(std::vector<uint8_t> const &bytes) : mBytes(bytes) {}
  physx::PxU32 read(void *destination, physx::PxU32 count) override {
    auto remaining = getLength() - mPosition;
    auto length = std::min(count, remaining);
    if (length) std::memcpy(destination, mBytes.data() + mPosition, length);
    mPosition += length;
    return length;
  }
  physx::PxU32 getLength() const override { return mBytes.size(); }
  void seek(physx::PxU32 position) override { mPosition = std::min(position, getLength()); }
  physx::PxU32 tell() const override { return mPosition; }
private:
  std::vector<uint8_t> const &mBytes;
  physx::PxU32 mPosition{};
};
}

namespace sapien::physx {
std::shared_ptr<PhysxConvexMesh>
PhysxConvexMesh::FromCookedData(std::vector<uint8_t> const &data) {
  if (data.empty() || data.size() > std::numeric_limits<::physx::PxU32>::max())
    throw std::invalid_argument("Cooked convex data must be nonempty and fit PxU32");
  auto mesh = std::shared_ptr<PhysxConvexMesh>(new PhysxConvexMesh());
  mesh->mEngine = PhysxEngine::Get();
  CookedInput input(data);
  mesh->mMesh = mesh->mEngine->getPxPhysics()->createConvexMesh(input);
  if (!mesh->mMesh) throw std::runtime_error("Failed to load cooked convex mesh");
  mesh->mAABB = computeAABB(mesh->getVertices());
  return mesh;
}
}
