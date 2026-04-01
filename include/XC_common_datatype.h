#ifndef _XC_COMMON_DATATYPE_H_
#define _XC_COMMON_DATATYPE_H_


//-------------------------------------------------------------------------------------------------
//  System Data Type
//-------------------------------------------------------------------------------------------------
/// data type unsigned char, data length 1 byte
typedef unsigned char XC_U8;
/// data type unsigned short, data length 2 bytes
typedef unsigned short XC_U16;
/// data type unsigned int, data length 4 bytes
typedef unsigned int XC_U32;
/// data type unsigned long long, data length 8 bytes
typedef unsigned long long XC_U64;
/// data type signed char, data length 1 byte

typedef signed char XC_S8;
/// data type signed short, data length 2 bytes
typedef signed short XC_S16;
/// data type signed int, data length 4 bytes
typedef signed int XC_S32;
/// data type signed long long, data length 8 bytes
typedef signed long long XC_S64;
/// data type float, data length 4 bytes

typedef float XC_FLOAT;
/// data type 64-bit physical address 8 bytes
typedef unsigned long long XC_PHY;
/// data type pointer content
typedef unsigned long XC_VIRT;

#endif