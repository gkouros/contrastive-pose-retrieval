import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from tqdm.notebook import tqdm
import imageio
import torch.nn as nn
import torch.nn.functional as F
from skimage import img_as_ubyte
from math import sqrt, atan2, degrees
import traceback

# io utils
try:
    from pytorch3d.io import load_obj  # io utils
    from pytorch3d.structures import Meshes  # datastructures
    from pytorch3d.transforms import Rotate, Translate  # 3D transformations
    from pytorch3d.renderer import ( # rendering components
        FoVPerspectiveCameras, OpenGLPerspectiveCameras, PerspectiveCameras,
        look_at_view_transform, look_at_rotation,
        RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
        SoftSilhouetteShader, HardPhongShader, HardFlatShader,
        SoftGouraudShader, HardGouraudShader, SoftPhongShader,
        PointLights, specular, TexturesVertex, FoVOrthographicCameras
    )
    from pytorch3d.ops import interpolate_face_attributes
    from pytorch3d.renderer.blending import softmax_rgb_blend
except ImportError:
    print('PyTorch3D could not be imported.')
    traceback.print_exc()


class Renderer:

    def __init__(self, device, intrinsics=None, image_size=(256, 256),
                 dataset='apolloscape'):
        self._device = device
        self._image_size = image_size

        if dataset == 'apolloscape':
            fov, aspect_ratio = self.calc_view_parameters(
                image_size, intrinsics)
            cameras = FoVPerspectiveCameras(device=self._device, fov=fov,
                                            aspect_ratio=aspect_ratio)
            # cameras = FoVOrthographicCameras(device=self._device, scale_xyz=((4,4,5),))
        elif dataset == 'pascal3d':
            # cameras = OpenGLPerspectiveCameras(device=device, fov=12.0)
            cameras = FoVPerspectiveCameras(device=self._device, fov=12,
                                            aspect_ratio=1)
        else:
            raise ValueError('Invalid dataset specified for renderer')
        self._blend_params = BlendParams(sigma=1e-6, gamma=1e-6)
        weight = np.log(1. / 1e-4 - 1.)

        # Create a silhouette mesh renderer with rasterizer and shader
        self._silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=RasterizationSettings(
                    image_size=self._image_size,
                    blur_radius=weight * self._blend_params.sigma,
                    faces_per_pixel=100,
                    # cull_backfaces=True,
                    max_faces_per_bin=100000,
                )
            ),
            shader=SoftSilhouetteShader(blend_params=self._blend_params)
        )
        # We can add a point light in front of the object.
        lights = PointLights(device=self._device, location=((-2, 4, 4),))

        # Create a phong renderer~
        self._phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=RasterizationSettings(
                    image_size=self._image_size,
                    blur_radius=weight * self._blend_params.sigma,
                    faces_per_pixel=100,
                    max_faces_per_bin=100000,
                )
            ),
            shader=SoftGouraudShader(device=self._device, cameras=cameras,
                                     lights=lights)
        )

        self._rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=RasterizationSettings(
                    image_size=self._image_size,
                    # high blur radius -> better differentiability
                    # low blur radius -> more accurate geometry
                    blur_radius=weight * self._blend_params.sigma,
                    faces_per_pixel=100,
                    perspective_correct=True,
                    cull_backfaces=True,
                    max_faces_per_bin=100000,
                )
        )

    def calc_view_parameters(self, image_size, intrinsics):
        if intrinsics is None:
            return (60, 1.0)
        h, w = image_size
        fx = intrinsics[0]
        fy = intrinsics[1]
        fovx = 2 * atan2(w, 2 * fx)
        fovy = 2 * atan2(h, 2 * fy)
        diag = sqrt(fovx ** 2 + fovy ** 2)
        fov = int(degrees(2 * atan2(diag, fovx + fovy)))
        aspect_ratio = w / h
        # return fov, aspect_ratio # fov is calculated as 70 but 60 works better
        # return (fov, aspect_ratio)
        return (60, aspect_ratio)

    def render(self, model, distance, elevation, azimuth, light_location=None):
        # Get the position of the camera based on the spherical angles
        R, T = look_at_view_transform(dist=distance,
                                      elev=elevation,
                                      azim=azimuth,
                                      device=self._device)
        return self.renderRT(model, R, T, light_location)

    def renderRT(self, model, R, T, light_location=None, type_='all',
                 return_arrays=True):
        # vertices are 3d points
        verts = torch.tensor(model['vertices']).to(torch.float32)
        # print(f'Vertices shape: {verts.shape}, vertices dtype: {verts.dtype}')

        # Each face consists 3 vertices
        faces = torch.tensor(np.array(model['faces']))

        # print(f'Faces shape: {faces.shape}, faces dtype: {faces.dtype}')

        # Initialize each vertex to be white in color.
        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(self._device))

        # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
        mesh = Meshes(
            verts=[verts.to(self._device)],
            faces=[faces.to(self._device)],
            textures=textures
        )

        if light_location is not None and len(light_location[0]) == 3:
            self._phong_renderer.shader.lights = PointLights(
                device=self._device, location=light_location)

        rendering = silhouette = depth = normals = None

        # generate rendering
        if type_ in ['rendering', 'all']:
            rendering = self._phong_renderer(
                meshes_world=mesh, R=R, T=T)[..., :3]  # 1 x N x M x 3
            if return_arrays:
                rendering = rendering.detach().cpu().numpy()

        # generate silhouette
        if type_ in ['silhouette', 'all']:
            # silhouette = rendering_to_silhouette(rendering)
            silhouette = self._silhouette_renderer(
                meshes_world=mesh, R=R, T=T)[..., 3]
            silhouette = torch.stack((silhouette,) * 3, axis=3)  # 1 x N x M x 3
            if return_arrays:
                silhouette = silhouette.detach().cpu().numpy()

        # get fragments for depth and normals
        if type_ in ['depth', 'normals', 'all']:
            fragments = self._rasterizer(mesh, R=R, T=T)

        # generate depth
        if type_ in ['depth', 'all']:
            # get depth
            depth = fragments.zbuf[..., 0]
            depth = torch.stack((depth,) * 3, axis=3)  # 1 x N x M x 3
            depth[depth >= 0] -= depth[depth >= 0].min()  # remove absolute depth
            depth[depth >= 0] /= depth.max()  # convert to 0-1 range
            if return_arrays:
                depth = depth.detach().cpu().numpy()

        # generate surface normals
        if type_ in ['normals', 'all']:
            normals = phong_normal_shading(mesh, fragments, self._blend_params)
            if return_arrays:
                normals = normals.detach().cpu().numpy()

        return rendering, silhouette, depth, normals


def rendering_to_silhouette(rendering):
    rendering = cv2.normalize(
        rendering, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    gray = cv2.cvtColor(255 - rendering, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)
    silhouette = np.zeros_like(rendering, dtype=np.uint8)
    cv2.drawContours(
        silhouette, contours, -1, (255, 255, 255), thickness=-1)
    silhouette = cv2.normalize(
        silhouette, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    return silhouette


def phong_normal_shading(meshes, fragments, blend_params):
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_normals = vertex_normals[faces]
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals)
    pixel_normals = 1-softmax_rgb_blend(pixel_normals, fragments, blend_params)
    return pixel_normals[..., :3]
