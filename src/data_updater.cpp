// File: DataUpdateNotifier.cpp
#include "data_updater.h"

std::mutex dataUpdateMutex;
std::condition_variable dataUpdateCV;
