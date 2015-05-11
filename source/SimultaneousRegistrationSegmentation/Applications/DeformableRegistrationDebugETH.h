extern "C" __declspec(dllexport) wchar_t* __cdecl DoRegistration(
  // Source volume
  int             sourceSizeX,
  int             sourceSizeY,   
  int             sourceSizeZ,
  double          sourceResX,
  double          sourceResY,
  double          sourceResZ,
  unsigned short* sourcePixels,
  int             sourceModality,
  double          sourceSlope,
  double          sourceIntercept,

  // Target volume
  int             targetSizeX,
  int             targetSizeY,
  int             targetSizeZ,
  double          targetResX,
  double          targetResY,
  double          targetResZ,
  unsigned short* targetPixels,
  int             targetModality,
  double          targetSlope,
  double          targetIntercept,
  
  // Rigid alignment
  double*         rigidAlignment,

  // VOI
  double             voiOriginX,
  double             voiOriginY,
  double             voiOriginZ,
  double             voiExtentX,
  double             voiExtentY,
  double             voiExtentZ,

  // Optional arguments for further extensions
  int             numOptionalArguments,
  wchar_t**       optionalArguments,

  // Output vector field
  int*            fieldSizeX,
  int*            fieldSizeY,
  int*            fieldSizeZ,
  double*         fieldResX,
  double*         fieldResY,
  double*         fieldResZ,
  double*         fieldOriginX,
  double*         fieldOriginY,
  double*         fieldOriginZ,
  float**         fieldVectors,
  
  // Progress callback function
  bool            (*progressCallbackFunc)(int progress)
  );

