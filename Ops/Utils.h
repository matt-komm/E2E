#ifndef VTX_OPS_UTILS_H
#define VTX_OPS_KERNELS_H

namespace utils
{

template<typename T>
inline T clamp(const T& v, const T& l, const T& h)
{
    return v < l ? l : (v > h ? h : v);
}

}

#endif
