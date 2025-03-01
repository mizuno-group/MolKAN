from setuptools import setup, find_packages

setup(
    name="clmpy",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "clmpy.gruvae.train=GRU_VAE.train:main",
            "clmpy.gruvae.evaluate=GRU_VAE.evaluate:main",
            "clmpy.gruvae.generate=GRU_VAE.generate:main",
            "clmpy.gruvae.encode=GRU_VAE.encode:main",
            "clmpy.gru.train=GRU.train:main",
            "clmpy.gru.evaluate=GRU.evaluate:main",
            "clmpy.gru.generate=GRU.generate:main",
            "clmpy.gru.encode=GRU.encode:main",
            "clmpy.transformerlatent.train=Transformer_latent.train:main",
            "clmpy.transformerlatent.evaluate=Transformer_latent.evaluate:main",
            "clmpy.transformerlatent.generate=Transformer_latent.generate:main",
            "clmpy.transformerlatent.encode=Transformer_latent.encode:main",
            "clmpy.transformervae.train=Transformer_VAE.train:main",
            "clmpy.transformervae.evaluate=Transformer_VAE.evaluate:main",
            "clmpy.transformervae.generate=Transformer_VAE.generate:main",
            "clmpy.transformervae.encode=Transformer_VAE.encode:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.12"
    ]
)