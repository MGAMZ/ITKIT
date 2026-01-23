"""
Test input/output shapes for neural network models in itkit.models.

SegFormer3D is the reference implementation with correct IO:
- Input: [B, C_in, D, H, W]
- Output: [B, C_out, D, H, W]

All models should follow this pattern for 3D models,
or [B, C_in, H, W] -> [B, C_out, H, W] for 2D models.
"""

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")

@pytest.mark.torch
def test_segformer3d_io():
    """Test SegFormer3D IO - reference implementation."""
    from itkit.models.SegFormer3D import SegFormer3D

    # Create model
    in_channels = 1
    num_classes = 3
    model = SegFormer3D(in_channels=in_channels, num_classes=num_classes)
    model.eval()

    # Test input
    batch_size = 2
    depth, height, width = 64, 64, 64
    x = torch.randn(batch_size, in_channels, depth, height, width)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    # Check output shape
    expected_shape = (batch_size, num_classes, depth, height, width)
    assert output.shape == expected_shape, \
        f"SegFormer3D output shape {output.shape} != expected {expected_shape}"

@pytest.mark.torch
@pytest.mark.parametrize("stage_count", [1, 2, 3, 4])
def test_segformer3d_variable_stage_counts(stage_count):
    """Test SegFormer3D IO with different stage counts."""
    from itkit.models.SegFormer3D import SegFormer3D

    in_channels = 1
    num_classes = 2

    embed_dims = [16, 32, 64, 128][:stage_count]
    num_heads = [1, 2, 4, 8][:stage_count]
    mlp_ratios = [2] * stage_count
    depths = [1] * stage_count
    sr_ratios = [1] * stage_count
    patch_kernel_size = [3] * stage_count
    patch_stride = [2] * stage_count
    patch_padding = [1] * stage_count

    model = SegFormer3D(
        in_channels=in_channels,
        embed_dims=embed_dims,
        num_heads=num_heads,
        mlp_ratios=mlp_ratios,
        depths=depths,
        sr_ratios=sr_ratios,
        patch_kernel_size=patch_kernel_size,
        patch_stride=patch_stride,
        patch_padding=patch_padding,
        num_classes=num_classes,
        decoder_head_embedding_dim=64,
    )
    model.eval()

    batch_size = 1
    depth, height, width = 32, 32, 32
    x = torch.randn(batch_size, in_channels, depth, height, width)

    with torch.no_grad():
        output = model(x)

    expected_shape = (batch_size, num_classes, depth, height, width)
    assert output.shape == expected_shape, \
        f"SegFormer3D({stage_count} stages) output shape {output.shape} != expected {expected_shape}"

@pytest.mark.torch
def test_unetr_io():
    """Test UNETR IO."""
    pytest.importorskip("monai", reason="MONAI not installed")
    from itkit.models.UNETR import UNETR

    # Create model with smaller size for testing
    in_channels = 1
    out_channels = 3
    img_size = (64, 64, 64)
    model = UNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        feature_size=16,
        hidden_size=256,
        mlp_dim=1024,
        num_heads=4,
        pos_embed="learnable",
        norm_name="instance",
        res_block=True,
    )
    model.eval()

    # Test input
    batch_size = 1
    depth, height, width = img_size
    x = torch.randn(batch_size, in_channels, depth, height, width)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    # Check output shape
    expected_shape = (batch_size, out_channels, depth, height, width)
    assert output.shape == expected_shape, \
        f"UNETR output shape {output.shape} != expected {expected_shape}"

@pytest.mark.torch
def test_mednext_io():
    """Test MedNeXt IO."""
    from itkit.models.MedNeXt import MedNeXt

    # Create model
    in_channels = 1
    n_classes = 3
    model = MedNeXt(
        in_channels=in_channels,
        n_channels=32,
        n_classes=n_classes,
        exp_r=2,
        kernel_size=3,
        deep_supervision=False,
        do_res=True,
        block_counts=[1, 1, 1, 1, 1, 1, 1, 1, 1],
        dim="3d",
    )
    model.eval()

    # Test input
    batch_size = 1
    depth, height, width = 64, 64, 64
    x = torch.randn(batch_size, in_channels, depth, height, width)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    # Check output shape
    expected_shape = (batch_size, n_classes, depth, height, width)
    assert output.shape == expected_shape, \
        f"MedNeXt output shape {output.shape} != expected {expected_shape}"

@pytest.mark.torch
def test_mednext_deep_supervision_io():
    """Test MedNeXt with deep supervision IO."""
    from itkit.models.MedNeXt import MedNeXt

    # Create model with deep supervision
    in_channels = 1
    n_classes = 3
    model = MedNeXt(
        in_channels=in_channels,
        n_channels=32,
        n_classes=n_classes,
        exp_r=2,
        kernel_size=3,
        deep_supervision=True,
        do_res=True,
        block_counts=[1, 1, 1, 1, 1, 1, 1, 1, 1],
        dim="3d",
    )
    model.eval()

    # Test input
    batch_size = 1
    depth, height, width = 64, 64, 64
    x = torch.randn(batch_size, in_channels, depth, height, width)

    # Forward pass
    with torch.no_grad():
        outputs = model(x)

    # With deep supervision, output is a list
    assert isinstance(outputs, list), "MedNeXt with deep_supervision should return a list"
    assert len(outputs) == 5, "MedNeXt with deep_supervision should return 5 outputs"

    # Check main output shape
    main_output = outputs[0]
    expected_shape = (batch_size, n_classes, depth, height, width)
    assert main_output.shape == expected_shape, \
        f"MedNeXt main output shape {main_output.shape} != expected {expected_shape}"

@pytest.mark.torch
def test_unet3plus_3d_io():
    """Test UNet3Plus 3D IO."""
    from itkit.models.UNet3Plus import UNet3Plus

    # Create 3D model
    in_channels = 1
    output_channels = 3
    depth, height, width = 64, 64, 64
    input_shape = [in_channels, depth, height, width]

    model = UNet3Plus(
        input_shape=input_shape,
        output_channels=output_channels,
        filters=[32, 64, 128, 256, 512],
        deep_supervision=False,
        ClassificationGuidedModule=False,
        dim=3,
    )
    model.eval()

    # Test input
    batch_size = 1
    x = torch.randn(batch_size, in_channels, depth, height, width)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    # Check output shape
    expected_shape = (batch_size, output_channels, depth, height, width)
    assert output.shape == expected_shape, \
        f"UNet3Plus 3D output shape {output.shape} != expected {expected_shape}"

@pytest.mark.torch
def test_unet3plus_2d_io():
    """Test UNet3Plus 2D IO."""
    from itkit.models.UNet3Plus import UNet3Plus

    # Create 2D model
    in_channels = 3
    output_channels = 1
    height, width = 256, 256
    input_shape = [in_channels, height, width]

    model = UNet3Plus(
        input_shape=input_shape,
        output_channels=output_channels,
        filters=[32, 64, 128, 256, 512],
        deep_supervision=False,
        ClassificationGuidedModule=False,
        dim=2,
    )
    model.eval()

    # Test input
    batch_size = 2
    x = torch.randn(batch_size, in_channels, height, width)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    # Check output shape
    expected_shape = (batch_size, output_channels, height, width)
    assert output.shape == expected_shape, \
        f"UNet3Plus 2D output shape {output.shape} != expected {expected_shape}"

@pytest.mark.torch
def test_segmamba_io():
    """Test SegMamba IO."""
    pytest.importorskip("mamba_ssm", reason="mamba_ssm not installed")
    pytest.importorskip("monai", reason="MONAI not installed")
    from itkit.models.SegMamba import SegMamba

    # Create model
    in_chans = 1
    out_chans = 3
    model = SegMamba(
        in_chans=in_chans,
        out_chans=out_chans,
        depths=[1, 1, 1, 1],
        feat_size=[48, 96, 192, 384],
        hidden_size=384,
        spatial_dims=3,
    )
    model.eval()

    # Test input - SegMamba typically works with 96^3 or 128^3
    batch_size = 1
    depth, height, width = 96, 96, 96
    x = torch.randn(batch_size, in_chans, depth, height, width)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    # Check output shape
    expected_shape = (batch_size, out_chans, depth, height, width)
    assert output.shape == expected_shape, \
        f"SegMamba output shape {output.shape} != expected {expected_shape}"

@pytest.mark.torch
def test_egeunet_io():
    """Test EGE-UNet IO (2D model)."""
    from itkit.models.EGE_UNet import EGEUNet

    # Create model
    num_classes = 1
    input_channels = 3
    model = EGEUNet(
        num_classes=num_classes,
        input_channels=input_channels,
        c_list=[8, 16, 24, 32, 48, 64],
        bridge=True,
        gt_ds=False,  # No deep supervision for simpler output
    )
    model.eval()

    # Test input (2D)
    batch_size = 2
    height, width = 256, 256
    x = torch.randn(batch_size, input_channels, height, width)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    # Check output shape
    expected_shape = (batch_size, num_classes, height, width)
    assert output.shape == expected_shape, \
        f"EGE-UNet output shape {output.shape} != expected {expected_shape}"

@pytest.mark.torch
def test_dconnnet_io():
    """Test DconnNet IO (2D model)."""
    from itkit.models.DconnNet.DconnNet import DconnNet

    # Create model
    in_chans = 3
    num_classes = 1
    model = DconnNet(in_chans=in_chans, num_classes=num_classes)
    model.eval()

    # Test input (2D)
    batch_size = 2
    height, width = 256, 256
    x = torch.randn(batch_size, in_chans, height, width)

    # Forward pass
    with torch.no_grad():
        output, _ = model(x)

    # DconnNet outputs num_classes*8 channels due to its architecture
    # Validate basic shape constraints
    assert output.shape[0] == batch_size, "Batch size mismatch"
    assert output.shape[1] == num_classes * 8, f"Expected {num_classes * 8} channels, got {output.shape[1]}"
    assert output.shape[2] == height, "Height mismatch"
    assert output.shape[3] == width, "Width mismatch"

@pytest.mark.torch
def test_smp_segformer_io():
    """Test SMP_Segformer IO (2D model using segmentation_models_pytorch)."""
    pytest.importorskip("segmentation_models_pytorch", reason="segmentation_models_pytorch not installed")
    from itkit.models.SMP_Segformer import SMP_Segformer

    # Create model
    in_channels = 3
    num_classes = 1
    model = SMP_Segformer(
        encoder_name="mit_b0",
        in_channels=in_channels,
        num_classes=num_classes,
    )
    model.eval()

    # Test input (2D)
    batch_size = 2
    height, width = 256, 256
    x = torch.randn(batch_size, in_channels, height, width)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    # Check output shape
    expected_shape = (batch_size, num_classes, height, width)
    assert output.shape == expected_shape, \
        f"SMP_Segformer output shape {output.shape} != expected {expected_shape}"

@pytest.mark.torch
def test_dsnet_io():
    """Test DSNet IO (2D model)."""
    pytest.importorskip("torchvision", reason="torchvision not installed")
    from itkit.models.DSNet import VGG16_DSNet

    # Create model
    model = VGG16_DSNet(logits_resize=None)
    model.eval()

    # Test input (2D)
    batch_size = 2
    in_channels = 3
    height, width = 224, 224
    x = torch.randn(batch_size, in_channels, height, width)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    # Check output - DSNet outputs single channel saliency map
    assert output.shape[0] == batch_size, "Batch size mismatch"
    assert output.shape[1] == 1, "DSNet should output 1 channel"

@pytest.mark.torch
def test_efficientnetv2_io():
    """Test EfficientNetV2 IO (segmentation model)."""
    pytest.importorskip("timm", reason="timm not installed")
    from itkit.models.EfficientNet import EfficientNetV2

    # Create model
    num_classes = 10
    model = EfficientNetV2(
        out_channels=num_classes,
        model_name="efficientnetv2_rw_s",
        in_chans=3,
    )
    model.eval()

    # Test input (2D images)
    batch_size = 2
    channels = 3
    height, width = 224, 224
    x = torch.randn(batch_size, channels, height, width)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    # Segmentation output should be [B, num_classes, H, W]
    expected_shape = (batch_size, num_classes, height, width)
    assert output.shape == expected_shape, \
        f"EfficientNetV2 output shape {output.shape} != expected {expected_shape}"

@pytest.mark.torch
def test_efficientformerv2_io():
    """Test EfficientFormerV2 IO (segmentation model)."""
    pytest.importorskip("timm", reason="timm not installed")
    from itkit.models.EfficientFormer import EfficientFormerV2

    # Create model
    num_classes = 10
    model = EfficientFormerV2(
        out_channels=num_classes,
        model_name="efficientformer_l1",
        in_chans=3,
    )
    model.eval()

    # Test input (2D images)
    batch_size = 2
    channels = 3
    height, width = 224, 224
    x = torch.randn(batch_size, channels, height, width)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    # Segmentation output should be [B, num_classes, H, W]
    expected_shape = (batch_size, num_classes, height, width)
    assert output.shape == expected_shape, \
        f"EfficientFormerV2 output shape {output.shape} != expected {expected_shape}"

@pytest.mark.torch
def test_datransunet_io():
    """Test DA-TransUNet IO (2D model)."""
    from itkit.models.DA_TransUnet.DATransUNet import DA_Transformer
    from itkit.models.DA_TransUnet.configs import get_r50_b16_config

    # Create model
    config = get_r50_b16_config()
    config.n_classes = 3
    img_size = 256
    model = DA_Transformer(config, img_size=img_size, num_classes=3)
    model.eval()

    # Test input (2D)
    batch_size = 1
    in_channels = 1  # DA_Transformer repeats 1 to 3 internally
    x = torch.randn(batch_size, in_channels, img_size, img_size)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    # Check output shape
    expected_shape = (batch_size, 3, img_size, img_size)
    assert output.shape == expected_shape, \
        f"DA-TransUNet output shape {output.shape} != expected {expected_shape}"

@pytest.mark.torch
def test_lmnet_io():
    """Test LM-Net IO (2D model)."""
    pytest.importorskip("natten", reason="LM-Net not installed")
    from itkit.models.LM_Net.LM_Net import MyUnet

    # Create model
    in_channels = 3
    num_classes = 2
    model = MyUnet(channel=in_channels, n_classes=num_classes, filters=[8, 16, 32, 64, 128])
    model.eval()

    # Test input (2D)
    batch_size = 1
    height, width = 224, 224
    x = torch.randn(batch_size, in_channels, height, width)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    # Check output shape
    expected_shape = (batch_size, num_classes, height, width)
    assert output.shape == expected_shape, \
        f"LM-Net output shape {output.shape} != expected {expected_shape}"

@pytest.mark.torch
def test_swinumamba_io():
    """Test SwinUMamba IO."""
    pytest.importorskip("mamba_ssm", reason="mamba_ssm not installed")
    from itkit.models.SwinUMamba import SwinUMambaD

    # Create model with minimal configuration
    in_chans = 1
    num_classes = 3
    depths = [1, 1, 1, 1]
    embed_dim = 48

    vss_args = {
        "in_chans": in_chans,
        "depths": depths,
        "dims": [embed_dim * (2**i) for i in range(len(depths))],
        "ssm_d_state": 1,
        "ssm_dt_rank": "auto",
        "ssm_ratio": 1.0,
        "mlp_ratio": 2.0,
        "downsample_version": "v3",
        "patchembed_version": "v2",
    }

    decoder_args = {
        "spatial_dims": 3,
        "in_channels": vss_args["dims"][-1],
        "skip_channels": vss_args["dims"],
        "num_classes": num_classes,
        "use_deconv": True,
    }

    model = SwinUMambaD(vss_args=vss_args, decoder_args=decoder_args)
    model.eval()

    # Test input
    batch_size = 1
    depth, height, width = 96, 96, 96
    x = torch.randn(batch_size, in_chans, depth, height, width)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    # Check output shape
    expected_shape = (batch_size, num_classes, depth, height, width)
    assert output.shape == expected_shape, \
        f"SwinUMamba output shape {output.shape} != expected {expected_shape}"

@pytest.mark.torch
def test_volumevssm_io():
    """Test VolumeVSSM IO."""
    pytest.importorskip("mamba_ssm", reason="mamba_ssm not installed")
    from itkit.models.VMamba.volume_mamba import VolumeVSSM, MambaAggregator1D

    # Mock backbone
    class MockBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dims = [48, 96, 192, 384]
        def forward(self, x):
            # x is [B*Z, C, H, W]
            # return [B*Z, 384, H//16, W//16]
            return torch.randn(x.shape[0], 384, x.shape[2]//16, x.shape[3]//16)

    backbone = MockBackbone()
    aggregator = MambaAggregator1D(d_model=384, out_dim=128)
    model = VolumeVSSM(slice_extractor_backbone=backbone, aggregator=aggregator)
    model.eval()

    # Test input
    batch_size = 1
    in_chans = 3
    depth, height, width = 16, 224, 224
    x = torch.randn(batch_size, in_chans, depth, height, width)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    # VolumeVSSM returns (vol_emb, )
    assert isinstance(output, tuple)
    assert output[0].shape == (batch_size, 128)


if __name__ == "__main__":
    # Run a basic test manually for debugging
    print("Running basic SegFormer3D test...")
    test_segformer3d_io()
    print("\nNote: This only tests SegFormer3D. Run pytest to test all models.")
