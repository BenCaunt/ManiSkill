"""Scene-local sphere visuals with stable particle identities across resets."""
import numpy as np
import sapien


class ParticleVisualPool:
    def __init__(self, scene):
        self.scene = scene
        self.entities, self.components, self.specs, self.active = [], [], [], []

    @staticmethod
    def model_specs(model):
        count = model.struct.n_particles
        colors = np.asarray(model.mpm_particle_colors[:count], dtype=np.float32)
        return [(model.struct.particle_radius, *color) for color in colors]

    def needs_change(self, model):
        specs = self.model_specs(model)
        return len(specs) != len(self.active) or any(old != new for old, new in zip(self.specs, specs))

    def configure(self, model):
        """Call only after releasing GPU groups if needs_change is true."""
        specs = self.model_specs(model)
        prototypes = {}
        for i, spec in enumerate(specs):
            if i < len(self.entities) and self.specs[i] == spec:
                self.components[i].visibility = 1.
                continue
            radius, *color = spec
            if spec not in prototypes:
                material = sapien.render.RenderMaterial(base_color=[*color, 1.], roughness=.8)
                prototypes[spec] = sapien.render.RenderShapeSphere(radius, material)
            component = sapien.render.RenderBodyComponent()
            component.attach(prototypes[spec].clone())
            if i < len(self.entities):
                entity = self.entities[i]
                entity.remove_component(self.components[i])
                entity.add_component(component)
                self.components[i], self.specs[i] = component, spec
            else:
                entity = sapien.Entity()
                entity.name = f'mpm_particle_visual_only_{i}'
                entity.add_component(component)
                self.scene.add_entity(entity)
                self.entities.append(entity); self.components.append(component); self.specs.append(spec)
        for component in self.components[len(specs):]:
            component.visibility = 0.
        self.active = self.entities[:len(specs)]

    def update(self, coupler):
        positions = coupler.states[0].struct.particle_q.numpy()[:len(self.active)]
        for entity, position in zip(self.active, positions):
            entity.pose = sapien.Pose(position)
        return positions
