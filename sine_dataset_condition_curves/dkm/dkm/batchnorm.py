import torch as t
import torch.nn as nn

from .kernels import SumKernel
from .lib import StructMat

############################
# Batchnorm parent classes #
############################

class InducingNormalisation(nn.Module):
    def forward(self, G):
        pass

class TrainTestNormalisation(nn.Module):
    def forward(self, G):
        pass

class InducingScaling(nn.Module):
    def __init__(self):
        super().__init__()

        self.initialised = False
        self.i_logscale = None

    def initialise(self, G):
        self.initialised = True

    def scale(self):
        return t.exp(self.i_logscale)

class TrainTestScaling(nn.Module):
    def __init__(self):
        super().__init__()

        self.initialised = False
        self.t_logscale = None
        
    def initialise(self, G):
        self.initialised = True

    def scale(self):
        return t.exp(self.t_logscale)

###################################
# Inducing Normalisation Children #
###################################

class InducingNormalisationNone(InducingNormalisation):
    def forward(self, G):
        return t.ones(1, device=G.ii.device)

class InducingNormalisationGlobal(InducingNormalisation):
    def forward(self, G):
        return t.sqrt(G.ii.diag().mean())

class InducingNormalisationLocal(InducingNormalisation):
    def forward(self, G):
        return t.sqrt(G.ii.diag())

#######################################
# Train / Test Normalisation Children #
#######################################

class TrainTestNormalisationNone(TrainTestNormalisation):
    def forward(self, G):
        return t.ones(1, device=G.ii.device)

class TrainTestNormalisationGlobal(TrainTestNormalisation):
    def forward(self, G):
        return t.sqrt(G.t.mean())

class TrainTestNormalisationLocal(TrainTestNormalisation):
    def forward(self, G):
        return t.sqrt(G.t)[...,None]

class TrainTestNormalisationLocation(TrainTestNormalisation):
    def forward(self, G):
        return t.sqrt(G.t.mean(0))[None,...,None]

class TrainTestNormalisationImage(TrainTestNormalisation):
    def forward(self, G):
        num_spatial_dims = G.t.dim() - 1
        if num_spatial_dims == 0:
            t_norm = t.sqrt(G.t)
        else:
            t_norm = t.sqrt(G.t.mean((1, num_spatial_dims)))
        return t_norm[[slice(None), *[None]*num_spatial_dims, None]] #Equivalent to [:, None, None, None] for a 2D image

#############################
# Inducing Scaling Children #
#############################

class InducingScalingNone(InducingScaling):
    def __init__(self):
        super().__init__()
        self.i_logscale = nn.Parameter(t.Tensor([0]), requires_grad=False)

class InducingScalingGlobal(InducingScaling):
    def __init__(self):
        super().__init__()
        self.i_logscale = nn.Parameter(t.Tensor([0]))

class InducingScalingLocal(InducingScaling):
    def __init__(self):
        super().__init__()

    def initialise(self, G):
        super().initialise(G)
        self.i_logscale = nn.Parameter(t.zeros(G.ii.shape[0], device=G.ii.device))

#################################
# Train / Test Scaling Children #
#################################

class TrainTestScalingNone(TrainTestScaling):
    def __init__(self):
        super().__init__()
        self.t_logscale = nn.Parameter(t.Tensor([0]), requires_grad=False)

class TrainTestScalingGlobal(TrainTestScaling):
    def __init__(self):
        super().__init__()
        self.t_logscale = nn.Parameter(t.Tensor([0]))

class TrainTestScalingLocation(TrainTestScaling):
    def __init__(self):
        super().__init__()

    def initialise(self, G):
        super().initialise(G)
        self.t_logscale = nn.Parameter(t.zeros(G.t.shape[1:G.t.dim()],device=G.ii.device)[None,...,None]) #Note that if there are no spatial dimensions, this will do nothing.

########################
# Main Batchnorm Class #
########################

class GramBatchNorm(nn.Module):

    INDUCING_NORMALISATION_CLASSES = {
        "none": InducingNormalisationNone,
        "global": InducingNormalisationGlobal,
        "local": InducingNormalisationLocal
    }

    TRAIN_TEST_NORMALISATION_CLASSES = {
        "none": TrainTestNormalisationNone,
        "global": TrainTestNormalisationGlobal,
        "local": TrainTestNormalisationLocal,
        "location": TrainTestNormalisationLocation,
        "image": TrainTestNormalisationImage
    }

    INDUCING_SCALING_CLASSES = {
        "none": InducingScalingNone,
        "global": InducingScalingGlobal,
        "local": InducingScalingLocal
    }

    TRAIN_TEST_SCALING_CLASSES = {
        "none": TrainTestScalingNone,
        "global": TrainTestScalingGlobal,
        "location": TrainTestScalingLocation
    }

    def __init__(self, i_norm="global", t_norm="global", i_scale="global", t_scale="global",
                 do_checkpointing=True):
        super().__init__()
        self.do_checkpointing = do_checkpointing
        self.i_norm = self.init_component(i_norm, self.INDUCING_NORMALISATION_CLASSES)
        self.t_norm = self.init_component(t_norm, self.TRAIN_TEST_NORMALISATION_CLASSES)
        self.i_scale = self.init_component(i_scale, self.INDUCING_SCALING_CLASSES)
        self.t_scale = self.init_component(t_scale, self.TRAIN_TEST_SCALING_CLASSES)

    @staticmethod
    def init_component(key, class_dict):
        if key not in class_dict:
            raise ValueError(f"Unknown key {key}")
        return class_dict[key]()

    def _forward(self, *G_tuple):
        G = StructMat(*G_tuple)

        if not self.i_scale.initialised:
            self.i_scale.initialise(G)
        if not self.t_scale.initialised:
            self.t_scale.initialise(G)

        #Norm and scale attributes already have appropriate broadcasting shapes
        i_norm = self.i_norm(G)
        t_norm = self.t_norm(G)
        i_scale = self.i_scale.scale()
        t_scale = self.t_scale.scale()

        #Normalise
        Gnorm_ii = G.ii / (i_norm * i_norm.unsqueeze(dim=-1))
        Gnorm_ti = G.ti / (i_norm * t_norm)
        Gnorm_t = G.t / t_norm.squeeze(dim=-1) ** 2

        #Scale
        Gnorm_ii = Gnorm_ii * (i_scale * i_scale[:,None])
        Gnorm_ti = Gnorm_ti * (t_scale * i_scale)
        Gnorm_t = Gnorm_t * t_scale.squeeze(dim=-1) ** 2

        return Gnorm_ii, Gnorm_ti, Gnorm_t

    def forward(self, G):
        if self.do_checkpointing:
            return StructMat(*t.utils.checkpoint.checkpoint(self._forward, *G, use_reentrant=False))
        else:
            return StructMat(*self._forward(*G))




###############
# OLD CLASSES #
###############



class GramBatchNorm1(nn.Module):
    def forward(self, G):
        a = (G.t.sum() + G.ii.diag().sum()) / (G.t.numel() + G.ii.shape[0])
        Gnorm_ii = G.ii / a
        Gnorm_ti = G.ti /a
        Gnorm_t = G.t / a
        return(StructMat(Gnorm_ii,Gnorm_ti,Gnorm_t, G.current_obj))

class GramBatchNorm1Scaled(nn.Module):
    def __init__(self):
        super().__init__()

        self.ii_logscale = nn.Parameter(t.Tensor([0]))
        self.tt_logscale = nn.Parameter(t.Tensor([0]))
    
    def forward(self, G):
        a = (G.t.sum() + G.ii.diag().sum()) / (G.t.numel() + G.ii.shape[0])
        Gnorm_ii = (G.ii / a) * t.exp(self.ii_logscale)**2
        Gnorm_ti = (G.ti /a) * t.exp(self.ii_logscale) * t.exp(self.tt_logscale)
        Gnorm_t = (G.t / a) * t.exp(self.tt_logscale)**2

        return(StructMat(Gnorm_ii,Gnorm_ti,Gnorm_t, G.current_obj))

class GramBatchNorm2(nn.Module):
    def forward(self, G):
        Gii_flat = G.ii
        Gti_flat = G.ti.reshape(-1,G.ti.shape[-1])
        gt_flat = G.t.view(-1)
        
        Gii_diag_sqrt = Gii_flat.diag().sqrt()
        gt_flat_sqrt = gt_flat.sqrt()
        
        Gii_flat_norm = Gii_flat / (Gii_diag_sqrt * Gii_diag_sqrt[:,None])
        Gti_flat_norm = Gti_flat / (Gii_diag_sqrt * gt_flat_sqrt[:,None])
        gt_flat_norm = t.ones(gt_flat.shape, device=gt_flat.device)
        
        Gii_norm = Gii_flat_norm.view(G.ii.shape)
        Gti_norm = Gti_flat_norm.reshape(G.ti.shape)
        gt_norm = gt_flat_norm.view(G.t.shape)
        
        return(StructMat(Gii_norm,Gti_norm,gt_norm, G.current_obj))

class GramBatchNorm2Scaled(nn.Module):
    def __init__(self):
        super().__init__()

        self.ii_logscale = nn.Parameter(t.Tensor([0]))
        self.tt_logscale = nn.Parameter(t.Tensor([0]))

    
    def forward(self, G):
        Gii_flat = G.ii
        Gti_flat = G.ti.reshape(-1,G.ti.shape[-1])
        gt_flat = G.t.view(-1)
        
        Gii_diag_sqrt = Gii_flat.diag().sqrt() * t.exp(self.ii_logscale)
        gt_flat_sqrt = gt_flat.sqrt() * t.exp(self.tt_logscale)
        
        Gii_flat_norm = Gii_flat / (Gii_diag_sqrt * Gii_diag_sqrt[:,None])
        Gti_flat_norm = Gti_flat / (Gii_diag_sqrt * gt_flat_sqrt[:,None])
        gt_flat_norm = gt_flat / (gt_flat_sqrt**2)
        
        Gii_norm = Gii_flat_norm.view(G.ii.shape)
        Gti_norm = Gti_flat_norm.reshape(G.ti.shape)
        gt_norm = gt_flat_norm.view(G.t.shape)
        
        return(StructMat(Gii_norm,Gti_norm,gt_norm, G.current_obj))

#Requires a forward pass to initialise the parameter tensors, before initialising the optimiser.
class GramBatchNorm2FineIndScaled(nn.Module):
    def __init__(self):
        super().__init__()

        self.ii_logscale = None
        self.tt_logscale = None
        self.initialised = False
    
    def forward(self, G):
        if not self.initialised:
            P = G.ii.shape[0]
            self.ii_logscale = nn.Parameter(t.Tensor([0] * P))
            self.tt_logscale = nn.Parameter(t.Tensor([0]))

        Gii_flat = G.ii
        Gti_flat = G.ti.reshape(-1,G.ti.shape[-1])
        gt_flat = G.t.view(-1)
        
        Gii_diag_sqrt = Gii_flat.diag().sqrt() * t.exp(self.ii_logscale)
        gt_flat_sqrt = gt_flat.sqrt() * t.exp(self.tt_logscale)
        
        Gii_flat_norm = Gii_flat / (Gii_diag_sqrt * Gii_diag_sqrt[:,None])
        Gti_flat_norm = Gti_flat / (Gii_diag_sqrt * gt_flat_sqrt[:,None])
        gt_flat_norm = gt_flat / gt_flat_sqrt**2
        
        Gii_norm = Gii_flat_norm.view(G.ii.shape)
        Gti_norm = Gti_flat_norm.reshape(G.ti.shape)
        gt_norm = gt_flat_norm.view(G.t.shape)
        
        return(StructMat(Gii_norm,Gti_norm,gt_norm, G.current_obj))


class GramBatchNorm2Location(nn.Module):
    def forward(self, G):
        Gii_diag_sqrt = G.ii.diag().sqrt()
        gt_avgspatial_sqrt = G.t.mean((0)).sqrt()

        Gii_norm = G.ii / (Gii_diag_sqrt * Gii_diag_sqrt[:, None])
        Gti_norm = G.ti / (Gii_diag_sqrt * gt_avgspatial_sqrt[None, :, :, None])
        gt_norm = G.t / (gt_avgspatial_sqrt[None, :, :] ** 2)

        breakpoint()

        return (StructMat(Gii_norm, Gti_norm, gt_norm, G.current_obj))


######################################################
# PIXEL AVERAGE CLASSES ONLY WORK WITH SPATIAL GRAMS #
######################################################
class GramBatchNorm2PixelAverage(nn.Module):
    def forward(self, G):
        Gii_diag_sqrt = G.ii.diag().sqrt()
        gt_avgspatial_sqrt = G.t.mean((1,2)).sqrt()
        
        Gii_norm =  G.ii / (Gii_diag_sqrt * Gii_diag_sqrt[:,None])
        Gti_norm = G.ti / (Gii_diag_sqrt * gt_avgspatial_sqrt[:,None,None,None])
        gt_norm = G.t / (gt_avgspatial_sqrt[:,None,None]**2)
        
        return(StructMat(Gii_norm, Gti_norm, gt_norm, G.current_obj))
        
class GramBatchNorm2PixelAverageScaled(nn.Module):
    def __init__(self):
        super().__init__()

        self.ii_logscale = nn.Parameter(t.Tensor([0]))
        self.tt_logscale = nn.Parameter(t.Tensor([0]))
        
    def forward(self, G):
        Gii_diag_sqrt = G.ii.diag().sqrt() * t.exp(self.ii_logscale)
        gt_avgspatial_sqrt = G.t.mean((1,2)).sqrt() * t.exp(self.tt_logscale)

        Gii_norm =  G.ii / (Gii_diag_sqrt * Gii_diag_sqrt[:,None])
        Gti_norm = G.ti / (Gii_diag_sqrt * gt_avgspatial_sqrt[:,None,None,None])
        gt_norm = G.t / (gt_avgspatial_sqrt[:,None,None]**2)
        
        return(StructMat(Gii_norm, Gti_norm, gt_norm, G.current_obj))

#Requires a forward pass to initialise the parameter tensors, before initialising the optimiser.
class GramBatchNorm2PixelAverageFineIndScaled(nn.Module):
    def __init__(self):
        super().__init__()

        self.ii_logscale = None
        self.tt_logscale = None
        self.initialised = False

        
    def forward(self, G):
        if not self.initialised:
            P = G.ii.shape[0]
            self.ii_logscale = nn.Parameter(t.Tensor([0] * P))
            self.tt_logscale = nn.Parameter(t.Tensor([0]))
            self.initialised = True

        Gii_diag_sqrt = G.ii.diag().sqrt() * t.exp(self.ii_logscale)
        gt_avgspatial_sqrt = G.t.mean((1,2)).sqrt() * t.exp(self.tt_logscale)

        Gii_norm =  G.ii / (Gii_diag_sqrt * Gii_diag_sqrt[:,None])
        Gti_norm = G.ti / (Gii_diag_sqrt * gt_avgspatial_sqrt[:,None,None,None])
        gt_norm = G.t / (gt_avgspatial_sqrt[:,None,None]**2)
        
        return(StructMat(Gii_norm, Gti_norm, gt_norm, G.current_obj))

#Requires a forward pass to initialise the parameter tensors, before initialising the optimiser.
class GramBatchNorm2PixelAverageFineIndPixelwiseScaled(nn.Module):
    def __init__(self):
        super().__init__()

        self.ii_logscale = None
        self.tt_logscale = None
        self.initialised = False
        
    def forward(self, G):
        if not self.initialised:
            P= G.ii.shape[0]
            width = G.t.shape[1]
            height = G.t.shape[2]

            self.ii_logscale = nn.Parameter(t.Tensor([0] * P))
            self.tt_logscale = nn.Parameter(t.Tensor([0]).repeat(width, height))

            self.initialised = True

        Gii_diag_sqrt = G.ii.diag().sqrt()
        gt_avgspatial_sqrt = G.t.mean((1,2)).sqrt()
        
        Gii_norm =  G.ii / (Gii_diag_sqrt * Gii_diag_sqrt[:,None])
        Gti_norm = G.ti / (Gii_diag_sqrt * gt_avgspatial_sqrt[:,None,None,None])
        gt_norm = G.t / (gt_avgspatial_sqrt[:,None,None]**2)
        
        ii_scale = t.exp(self.ii_logscale)
        tt_scale = t.exp(self.tt_logscale)
        
        Gii_rescaled =  Gii_norm * ii_scale * ii_scale[:,None]
        Gti_rescaled = Gti_norm * ii_scale * tt_scale[None,:,:,None]
        gt_rescaled = gt_norm * tt_scale[None,:,:]**2
        
        return(StructMat(Gii_rescaled, Gti_rescaled, gt_rescaled, G.current_obj))
