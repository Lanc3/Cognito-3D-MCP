# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from setuptools import find_packages, setup

setup(
    name="hy3dgen",
    version="2.0.2+mcp.1",
    description="Cognito-3D-MCP: local multi-backend image-to-3D generation for MCP clients",
    url="https://github.com/Lanc3/Cognito-3D-MCP",
    project_urls={
        "Upstream": "https://github.com/Tencent-Hunyuan/Hunyuan3D-2",
        "Issues": "https://github.com/Lanc3/Cognito-3D-MCP/issues",
    },
    license_files=("LICENSE", "NOTICE"),
    packages=find_packages(),
    include_package_data=True,
    package_data={"hy3dgen": ["assets/*", "assets/**/*"]},
    python_requires=">=3.10",
    extras_require={
        "mcp": ["mcp>=1.28,<2"],
        "test": [],
    },
    entry_points={
        "console_scripts": [
            "cognito-3d-mcp=hy3dgen_mcp.server:main",
            "hunyuan3d-mcp=hy3dgen_mcp.server:main",
        ],
    },
    install_requires=[
        'gradio',
        "tqdm>=4.66.3",
        'numpy',
        'ninja',
        'diffusers',
        'pybind11',
        'opencv-python',
        'einops',
        "transformers>=4.48.0",
        'omegaconf',
        'trimesh',
        'pymeshlab',
        'pygltflib',
        'xatlas',
        'accelerate',
        'fastapi',
        'uvicorn',
        'rembg',
        'onnxruntime'
    ]
)
