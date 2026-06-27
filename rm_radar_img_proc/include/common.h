
#pragma once

namespace rm_radarplugin
{
enum class ArmorColor
{
  RED = 0,
  BLUE = 1,
  OTHER = 2,
  WHITE = 3
};

enum class ArmorTargetType
{
  ALL = 0,
  OUTPOST_BASE = 1,
  WITHOUT_OUTPOST_BASE = 2
};

enum class DrawImage
{
  DISABLE = 0,
  RAW = 1,
  BINARY = 2,
  MORPHOLOGY = 3,
  BARS = 4,
  ARMORS = 5,
  ARMORS_VERTEXS = 6,
  BARS_ARMORS = 7,
  WARP = 8,
  PROJECT = 9
};

enum PreProcessMethod
{
  HSV = 0,
  SINGLE_CHANNEL = 1
};

enum DetectMethod
{
  TRADITIONAL = 0,
  YOLO = 1
};

enum MorphType
{
  ERODE = 0,
  DILATE = 1,
  OPEN = 2,
  CLOSE = 3,
  GRADIENT = 4,
  TOPHAT = 5,
  BLACKHAT = 6,
  HITMISS = 7,
  DISABLE = 8
};

enum MorphKernelShape
{
  ELLIPSE = 0,
  RECT = 1,
  LINE = 2
};

enum CamType
{
  Tele_cam = 0,
  Wide_cam = 1
};


}  // namespace rm_digtialimg_proc
