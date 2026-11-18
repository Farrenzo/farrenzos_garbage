from .fg_paint_latent import FG_MiniMaxH3PaintLatent
from .fg_minimax_h3 import FG_MiniMaxH3_Conditioner
from .fg_video_resolution import (
    FG_VideoSuperResolution,
    FG_VideoFrameGeneration,
    FG_VideoHDR,
)
from .fg_minimax_autochain import (
    MiniMaxH3AutoChainMotionContext,
    MiniMaxH3AutoChainMotionContextTrim,
    MiniMaxH3AutoChainSaveLatent,
    MiniMaxH3AutoChainLoadLatent,
    MiniMaxH3AutoChainAudio,
    MiniMaxH3AutoChain,
    MiniMaxH3AutoChainFrameReference,
)

MINIMAX_NODE_CLASS_MAPPINGS = {
    "FG_MiniMaxH3PaintLatent"            : FG_MiniMaxH3PaintLatent,
    "FG_MiniMaxH3_Conditioner"           : FG_MiniMaxH3_Conditioner,
    "MiniMaxH3AutoChain"                 : MiniMaxH3AutoChain,
    "MiniMaxH3AutoChainAudio"            : MiniMaxH3AutoChainAudio,
    "MiniMaxH3AutoChainSaveLatent"       : MiniMaxH3AutoChainSaveLatent,
    "MiniMaxH3AutoChainLoadLatent"       : MiniMaxH3AutoChainLoadLatent,
    "MiniMaxH3AutoChainFrameReference"   : MiniMaxH3AutoChainFrameReference,
    "MiniMaxH3AutoChainMotionContext"    : MiniMaxH3AutoChainMotionContext,
    "MiniMaxH3AutoChainMotionContextTrim": MiniMaxH3AutoChainMotionContextTrim,

    "FG_VideoHDR": FG_VideoHDR,
    "FG_VideoSuperResolution": FG_VideoSuperResolution,
    "FG_VideoFrameGeneration": FG_VideoFrameGeneration,
}
MINIMAX_NODE_DISPLAY_NAME_MAPPINGS = {
    "FG_MiniMaxH3PaintLatent": "⚙️ Paint Keyframes Into MiniMax H3 Latent",
    "FG_MiniMaxH3_Conditioner"      : "⚙️ MiniMax H3 Conditioner",
    "MiniMaxH3AutoChainMotionContext":     "⚙️ H3 Auto Chain Motion Context",
    "MiniMaxH3AutoChainMotionContextTrim": "⚙️ H3 Auto Chain Motion Context Trim",
    "MiniMaxH3AutoChainSaveLatent":        "⚙️ H3 Auto Chain Save Latent",
    "MiniMaxH3AutoChainLoadLatent":        "⚙️ H3 Auto Chain Load Latent",
    "MiniMaxH3AutoChainAudio":             "⚙️ H3 Auto Chain Audio",
    "MiniMaxH3AutoChain":                  "⚙️ H3 Auto Chain + Stitch",
    "MiniMaxH3AutoChainFrameReference":    "⚙️ H3 Auto Chain Frame Reference",

    "FG_VideoHDR": "⚙️ Video SDR to HDR (Any Device)",
    "FG_VideoSuperResolution": "⚙️ Video Super Resolution (Any Device)",
    "FG_VideoFrameGeneration": "⚙️ Video Frame Generation (Any Device)",
}