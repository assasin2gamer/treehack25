// File: DataUpdateNotifier.h
#ifndef DATA_UPDATER_H
#define DATA_UPDATER_H

#include <mutex>
#include <condition_variable>

extern std::mutex dataUpdateMutex;
extern std::condition_variable dataUpdateCV;

#endif // DATA_UPDATER_H
