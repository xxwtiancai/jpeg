/**
 * © 2013, 2015 Kornel Lesiński. All rights reserved.
 * Based on code by Rich Geldreich.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "jpge.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <immintrin.h>  // AVX/AVX2
#include <math.h>
// 添加SSE2指令集头文件
#if defined(__SSE2__) || defined(_MSC_VER)
#include <emmintrin.h>
#define SUPPORT_SSE2
#endif

// 定义最大值和最小值宏
#define JPGE_MAX(a,b) (((a)>(b))?(a):(b))
#define JPGE_MIN(a,b) (((a)<(b))?(a):(b))
const double PI = 3.14159265358979323846;

namespace jpge
{

/**
 * 分配内存函数
 * 封装了标准库的malloc函数
 */
static inline void *jpge_malloc(size_t nSize)
{
    return malloc(nSize); // 调用标准库的malloc函数分配nSize字节的内存
}

/**
 * 释放内存函数
 * 封装了标准库的free函数
 */
static inline void jpge_free(void *p)
{
    free(p); // 调用标准库的free函数释放内存
}

// JPEG相关的枚举常量和表格
enum { M_SOF0 = 0xC0, M_DHT = 0xC4, M_SOI = 0xD8, M_EOI = 0xD9, M_SOS = 0xDA, M_DQT = 0xDB, M_APP0 = 0xE0 }; // JPEG标记
enum { DC_LUM_CODES = 12, AC_LUM_CODES = 256, DC_CHROMA_CODES = 12, AC_CHROMA_CODES = 256, MAX_HUFF_SYMBOLS = 257, MAX_HUFF_CODESIZE = 32 }; // Huffman编码相关常量

// zigzag扫描顺序表，用于将8x8块中的系数按频率排序
static uint8 s_zag[64] = { 0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63 };

// 标准亮度量化表
static int16 s_std_lum_quant[64] = { 16,11,12,14,12,10,16,14,13,14,18,17,16,19,24,40,26,24,22,22,24,49,35,37,29,40,58,51,61,60,57,51,56,55,64,72,92,78,64,68,87,69,55,56,80,109,81,87,95,98,103,104,103,62,77,113,121,112,100,120,92,101,103,99 };

// 标准色度量化表
static int16 s_std_croma_quant[64] = { 17,18,18,24,21,24,47,26,26,47,99,66,56,66,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99 };

/**
 * 清空对象函数
 * 使用memset将对象的内存设为0
 */
template <class T> inline void clear_obj(T &obj)
{
    memset(&obj, 0, sizeof(obj)); // 将对象的所有字节清零
}

/**
 * RGB转YCC颜色空间转换函数
 * 将RGB像素转换为YCbCr颜色空间
 */
template<class T> static void RGB_to_YCC(image *img, const T *src, int width, int y)
{
    for (int x = 0; x < width; x++) {
        const int r = src[x].r, g = src[x].g, b = src[x].b; // 获取RGB值
        img[0].set_px((0.299f * r + 0.587f * g + 0.114f * b) - 128.0f, x, y); // Y分量
        img[1].set_px((-0.168736f * r - 0.331264f * g + 0.5f * b), x, y); // Cb分量
        img[2].set_px((0.5f * r - 0.418688f * g - 0.081312f * b), x, y); // Cr分量
    }
}

/**
 * RGB转YCC颜色空间转换函数(AVX2优化版)
 * 使用AVX2指令集优化的RGB到YCbCr的转换
 */
template<class T> static void RGB_to_YCC_AVX2(image *img, const T *src, int width, int y)
{
    // RGB到YCC转换系数作为SIMD向量
    const __m256 y_coeffs = _mm256_setr_ps(0.299f, 0.587f, 0.114f, 0.0f, 0.299f, 0.587f, 0.114f, 0.0f);
    const __m256 cb_coeffs = _mm256_setr_ps(-0.168736f, -0.331264f, 0.5f, 0.0f, -0.168736f, -0.331264f, 0.5f, 0.0f);
    const __m256 cr_coeffs = _mm256_setr_ps(0.5f, -0.418688f, -0.081312f, 0.0f, 0.5f, -0.418688f, -0.081312f, 0.0f);
    const __m256 y_offset = _mm256_set1_ps(128.0f);
    
    int x = 0;
    
    // 每次处理8个像素
    for (; x + 7 < width; x += 8) {
        // 加载8个像素的R, G, B值
        __m256i r_i_low = _mm256_setr_epi32(
            src[x].r, src[x+1].r, src[x+2].r, src[x+3].r,
            src[x+4].r, src[x+5].r, src[x+6].r, src[x+7].r
        );
        __m256i g_i_low = _mm256_setr_epi32(
            src[x].g, src[x+1].g, src[x+2].g, src[x+3].g,
            src[x+4].g, src[x+5].g, src[x+6].g, src[x+7].g
        );
        __m256i b_i_low = _mm256_setr_epi32(
            src[x].b, src[x+1].b, src[x+2].b, src[x+3].b,
            src[x+4].b, src[x+5].b, src[x+6].b, src[x+7].b
        );
        
        // 转换为浮点数
        __m256 r = _mm256_cvtepi32_ps(r_i_low);
        __m256 g = _mm256_cvtepi32_ps(g_i_low);
        __m256 b = _mm256_cvtepi32_ps(b_i_low);
        
        // 计算Y分量(亮度)
        __m256 r_y = _mm256_mul_ps(r, _mm256_shuffle_ps(y_coeffs, y_coeffs, _MM_SHUFFLE(0, 0, 0, 0)));
        __m256 g_y = _mm256_mul_ps(g, _mm256_shuffle_ps(y_coeffs, y_coeffs, _MM_SHUFFLE(1, 1, 1, 1)));
        __m256 b_y = _mm256_mul_ps(b, _mm256_shuffle_ps(y_coeffs, y_coeffs, _MM_SHUFFLE(2, 2, 2, 2)));
        __m256 y_val = _mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(r_y, g_y), b_y), y_offset);
        
        // 计算Cb分量(蓝色差)
        __m256 r_cb = _mm256_mul_ps(r, _mm256_shuffle_ps(cb_coeffs, cb_coeffs, _MM_SHUFFLE(0, 0, 0, 0)));
        __m256 g_cb = _mm256_mul_ps(g, _mm256_shuffle_ps(cb_coeffs, cb_coeffs, _MM_SHUFFLE(1, 1, 1, 1)));
        __m256 b_cb = _mm256_mul_ps(b, _mm256_shuffle_ps(cb_coeffs, cb_coeffs, _MM_SHUFFLE(2, 2, 2, 2)));
        __m256 cb_val = _mm256_add_ps(_mm256_add_ps(r_cb, g_cb), b_cb);
        
        // 计算Cr分量(红色差)
        __m256 r_cr = _mm256_mul_ps(r, _mm256_shuffle_ps(cr_coeffs, cr_coeffs, _MM_SHUFFLE(0, 0, 0, 0)));
        __m256 g_cr = _mm256_mul_ps(g, _mm256_shuffle_ps(cr_coeffs, cr_coeffs, _MM_SHUFFLE(1, 1, 1, 1)));
        __m256 b_cr = _mm256_mul_ps(b, _mm256_shuffle_ps(cr_coeffs, cr_coeffs, _MM_SHUFFLE(2, 2, 2, 2)));
        __m256 cr_val = _mm256_add_ps(_mm256_add_ps(r_cr, g_cr), b_cr);
        
        // 存储结果
        float y_results[8], cb_results[8], cr_results[8];
        _mm256_storeu_ps(y_results, y_val);
        _mm256_storeu_ps(cb_results, cb_val);
        _mm256_storeu_ps(cr_results, cr_val);
        
        // 设置像素值
        for (int i = 0; i < 8; i++) {
            img[0].set_px(y_results[i], x + i, y);
            img[1].set_px(cb_results[i], x + i, y);
            img[2].set_px(cr_results[i], x + i, y);
        }
    }
    
    // 处理剩余像素
    for (; x < width; x++) {
        const int r = src[x].r, g = src[x].g, b = src[x].b;
        img[0].set_px((0.299f * r + 0.587f * g + 0.114f * b) - 128.0f, x, y);
        img[1].set_px((-0.168736f * r - 0.331264f * g + 0.5f * b), x, y);
        img[2].set_px((0.5f * r - 0.418688f * g - 0.081312f * b), x, y);
    }
}

/**
 * RGB转Y函数
 * 将RGB像素转换为仅Y(亮度)分量，用于灰度图像
 */
template<class T> static void RGB_to_Y(image &img, const T *pSrc, int width, int y)
{
    for (int x=0; x < width; x++) {
        img.set_px((pSrc[x].r*0.299f) + (pSrc[x].g*0.587f) + (pSrc[x].b*0.114f) - 128.0f, x, y); // 计算亮度并减去128
    }
}

/**
 * RGB转Y函数(AVX优化版)
 * 使用AVX指令集优化的RGB到Y转换
 */
template<class T> static void RGB_to_Y_AVX(image &img, const T *pSrc, int width, int y) {
    // RGB到Y转换的常量
    const __m256 coeffs = _mm256_setr_ps(0.299f, 0.587f, 0.114f, 0.0f, 0.299f, 0.587f, 0.114f, 0.0f);
    const __m256 offset = _mm256_set1_ps(128.0f);
    
    int x = 0;
    
    // 每次处理8个像素
    for (; x + 7 < width; x += 8) {
        // 加载8个像素
        __m256 r1 = _mm256_setr_ps(
            pSrc[x].r, pSrc[x+1].r, pSrc[x+2].r, pSrc[x+3].r,
            pSrc[x+4].r, pSrc[x+5].r, pSrc[x+6].r, pSrc[x+7].r
        );
        __m256 g1 = _mm256_setr_ps(
            pSrc[x].g, pSrc[x+1].g, pSrc[x+2].g, pSrc[x+3].g,
            pSrc[x+4].g, pSrc[x+5].g, pSrc[x+6].g, pSrc[x+7].g
        );
        __m256 b1 = _mm256_setr_ps(
            pSrc[x].b, pSrc[x+1].b, pSrc[x+2].b, pSrc[x+3].b,
            pSrc[x+4].b, pSrc[x+5].b, pSrc[x+6].b, pSrc[x+7].b
        );
        
        // 应用系数(如果有FMA指令会更高效)
        __m256 rCoeff = _mm256_mul_ps(r1, _mm256_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(0, 0, 0, 0)));
        __m256 gCoeff = _mm256_mul_ps(g1, _mm256_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(1, 1, 1, 1)));
        __m256 bCoeff = _mm256_mul_ps(b1, _mm256_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(2, 2, 2, 2)));
        
        // 求和并减去偏移量
        __m256 sum = _mm256_add_ps(_mm256_add_ps(rCoeff, gCoeff), bCoeff);
        __m256 result = _mm256_sub_ps(sum, offset);
        
        // 存储结果
        float values[8];
        _mm256_storeu_ps(values, result);
        
        for (int i = 0; i < 8; i++) {
            img.set_px(values[i], x+i, y);
        }
    }
    
    // 处理剩余像素
    for (; x < width; x++) {
        img.set_px((pSrc[x].r*0.299f) + (pSrc[x].g*0.587f) + (pSrc[x].b*0.114f) - 128.0f, x, y);
    }
}

/**
 * Y转YCC函数
 * 将灰度图像(仅Y)转换为YCbCr格式，Cb和Cr设为0
 */
static void Y_to_YCC(image *img, const uint8 *pSrc, int width, int y)
{
    for(int x=0; x < width; x++) {
        img[0].set_px(pSrc[x]-128.0, x, y); // Y分量，减去128
        img[1].set_px(0, x, y); // Cb分量设为0
        img[2].set_px(0, x, y); // Cr分量设为0
    }
}

/**
 * 获取像素值
 * 获取图像中指定位置的像素值
 */
inline float image::get_px(int x, int y)
{
    return m_pixels[y*m_x + x]; // 计算像素在一维数组中的索引并返回值
}

/**
 * 设置像素值
 * 设置图像中指定位置的像素值
 */
inline void image::set_px(float px, int x, int y)
{
    m_pixels[y*m_x + x] = px; // 计算像素在一维数组中的索引并设置值
}

/**
 * 获取DCT量化系数块
 * 返回指定位置8x8块的量化DCT系数指针
 */
dctq_t *image::get_dctq(int x, int y)
{
    return &m_dctqs[64*(y/8 * m_x/8 + x/8)]; // 计算8x8块的索引位置
}

/**
 * 图像子采样函数
 * 对色度通道进行降采样以减少数据量
 */
void image::subsample(image &luma, int v_samp)
{
    if (v_samp == 2) {
        // 2x2降采样：水平和垂直方向都减半
        for(int y=0; y < m_y; y+=2) {
            for(int x=0; x < m_x; x+=2) {
                m_pixels[m_x/4*y + x/2] = blend_quad(x, y, luma); // 对2x2块计算平均值
            }
        }
        m_x /= 2; // 降采样后宽度减半
        m_y /= 2; // 降采样后高度减半
    } else {
        // 2x1降采样：只在水平方向减半
        for(int y=0; y < m_y; y++) {
            for(int x=0; x < m_x; x+=2) {
                m_pixels[m_x/2*y + x/2] = blend_dual(x, y, luma); // 对1x2块计算平均值
            }
        }
        m_x /= 2; // 降采样后宽度减半
    }
}

/**
 * 离散余弦变换函数
 * 对8x8像素块进行DCT变换，将空间域转换为频率域
 */
static void dct(dct_t *data)
{
    dct_t z1, z2, z3, z4, z5, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp10, tmp11, tmp12, tmp13, *data_ptr;

    data_ptr = data;

    // 对每一行进行DCT变换
    for (int c=0; c < 8; c++) {
        tmp0 = data_ptr[0] + data_ptr[7]; // 先计算一些可重用的值
        tmp7 = data_ptr[0] - data_ptr[7];
        tmp1 = data_ptr[1] + data_ptr[6];
        tmp6 = data_ptr[1] - data_ptr[6];
        tmp2 = data_ptr[2] + data_ptr[5];
        tmp5 = data_ptr[2] - data_ptr[5];
        tmp3 = data_ptr[3] + data_ptr[4];
        tmp4 = data_ptr[3] - data_ptr[4];
        
        // 偶数部分的处理
        tmp10 = tmp0 + tmp3;
        tmp13 = tmp0 - tmp3;
        tmp11 = tmp1 + tmp2;
        tmp12 = tmp1 - tmp2;
        
        data_ptr[0] = tmp10 + tmp11; // 最终结果写回到原数组
        data_ptr[4] = tmp10 - tmp11;
        
        z1 = (tmp12 + tmp13) * 0.541196100;
        data_ptr[2] = z1 + tmp13 * 0.765366865;
        data_ptr[6] = z1 + tmp12 * - 1.847759065;
        
        // 奇数部分的处理(更复杂)
        z1 = tmp4 + tmp7;
        z2 = tmp5 + tmp6;
        z3 = tmp4 + tmp6;
        z4 = tmp5 + tmp7;
        z5 = (z3 + z4) * 1.175875602;
        
        tmp4 *= 0.298631336;
        tmp5 *= 2.053119869;
        tmp6 *= 3.072711026;
        tmp7 *= 1.501321110;
        
        z1 *= -0.899976223;
        z2 *= -2.562915447;
        z3 *= -1.961570560;
        z4 *= -0.390180644;
        
        z3 += z5;
        z4 += z5;
        
        data_ptr[7] = tmp4 + z1 + z3;
        data_ptr[5] = tmp5 + z2 + z4;
        data_ptr[3] = tmp6 + z2 + z3;
        data_ptr[1] = tmp7 + z1 + z4;
        
        data_ptr += 8; // 移动到下一行
    }

    data_ptr = data;

    // 对每一列进行DCT变换
    for (int c=0; c < 8; c++) {
        tmp0 = data_ptr[8*0] + data_ptr[8*7];
        tmp7 = data_ptr[8*0] - data_ptr[8*7];
        tmp1 = data_ptr[8*1] + data_ptr[8*6];
        tmp6 = data_ptr[8*1] - data_ptr[8*6];
        tmp2 = data_ptr[8*2] + data_ptr[8*5];
        tmp5 = data_ptr[8*2] - data_ptr[8*5];
        tmp3 = data_ptr[8*3] + data_ptr[8*4];
        tmp4 = data_ptr[8*3] - data_ptr[8*4];
        
        tmp10 = tmp0 + tmp3;
        tmp13 = tmp0 - tmp3;
        tmp11 = tmp1 + tmp2;
        tmp12 = tmp1 - tmp2;
        
        // 最后应用1/8的缩放因子
        data_ptr[8*0] = (tmp10 + tmp11) / 8.0;
        data_ptr[8*4] = (tmp10 - tmp11) / 8.0;
        
        z1 = (tmp12 + tmp13) * 0.541196100;
        data_ptr[8*2] = (z1 + tmp13 * 0.765366865) / 8.0;
        data_ptr[8*6] = (z1 + tmp12 * -1.847759065) / 8.0;
        
        z1 = tmp4 + tmp7;
        z2 = tmp5 + tmp6;
        z3 = tmp4 + tmp6;
        z4 = tmp5 + tmp7;
        z5 = (z3 + z4) * 1.175875602;
        
        tmp4 *= 0.298631336;
        tmp5 *= 2.053119869;
        tmp6 *= 3.072711026;
        tmp7 *= 1.501321110;
        
        z1 *= -0.899976223;
        z2 *= -2.562915447;
        z3 *= -1.961570560;
        z4 *= -0.390180644;
        
        z3 += z5;
        z4 += z5;
        
        data_ptr[8*7] = (tmp4 + z1 + z3) / 8.0;
        data_ptr[8*5] = (tmp5 + z2 + z4) / 8.0;
        data_ptr[8*3] = (tmp6 + z2 + z3) / 8.0;
        data_ptr[8*1] = (tmp7 + z1 + z4) / 8.0;
        
        data_ptr++; // 移动到下一列
    }
}

/**
 * 符号频率结构体
 * 用于Huffman编码过程中记录符号及其出现频率
 */
struct sym_freq {
    uint m_key, m_sym_index; // m_key存储符号频率，m_sym_index存储符号索引
};

/**
 * 基数排序函数
 * 对符号频率数组按照频率(m_key)进行基数排序
 * @param num_syms 符号数量
 * @param pSyms0 输入符号数组
 * @param pSyms1 辅助排序数组
 * @return 指向排序后结果的指针
 */
static inline sym_freq *radix_sort_syms(uint num_syms, sym_freq *pSyms0, sym_freq *pSyms1)
{
    const uint cMaxPasses = 4; // 最大排序轮数，对应32位整数的4个字节
    uint32 hist[256 * cMaxPasses]; clear_obj(hist); // 初始化直方图数组为0
    
    // 统计每个位置每个值的出现次数
    for (uint i = 0; i < num_syms; i++) {
        uint freq = pSyms0[i].m_key;
        hist[freq & 0xFF]++; // 第1字节
        hist[256 + ((freq >> 8) & 0xFF)]++; // 第2字节
        hist[256*2 + ((freq >> 16) & 0xFF)]++; // 第3字节
        hist[256*3 + ((freq >> 24) & 0xFF)]++; // 第4字节
    }
    
    sym_freq *pCur_syms = pSyms0, *pNew_syms = pSyms1; // 当前和新的符号数组指针
    uint total_passes = cMaxPasses; // 总排序轮数
    
    // 优化：如果高位字节所有值都相同，可以减少排序轮数
    while ((total_passes > 1) && (num_syms == hist[(total_passes - 1) * 256])) {
        total_passes--;
    }
    
    // 对每个字节进行基数排序
    for (uint pass_shift = 0, pass = 0; pass < total_passes; pass++, pass_shift += 8) {
        const uint32 *pHist = &hist[pass << 8]; // 当前轮次的直方图指针
        uint offsets[256], cur_ofs = 0; // 计算每个值在输出数组中的起始位置
        
        for (uint i = 0; i < 256; i++) {
            offsets[i] = cur_ofs; // 记录当前偏移
            cur_ofs += pHist[i]; // 累加计算下一个值的偏移
        }
        
        // 按当前字节的值排序
        for (uint i = 0; i < num_syms; i++) {
            pNew_syms[offsets[(pCur_syms[i].m_key >> pass_shift) & 0xFF]++] = pCur_syms[i];
        }
        
        // 交换当前和新的数组指针
        sym_freq *t = pCur_syms;
        pCur_syms = pNew_syms;
        pNew_syms = t;
    }
    
    return pCur_syms; // 返回排序后的数组指针
}

/**
 * 计算最小冗余编码长度
 * 基于符号频率生成最优前缀编码长度
 * 原作者: Alistair Moffat, Jyrki Katajainen (1996)
 */
static void calculate_minimum_redundancy(sym_freq *A, int n)
{
    int root, leaf, next, avbl, used, dpth;
    
    if (n==0) { // 边界情况：没有符号
        return;
    } else if (n==1) { // 边界情况：只有一个符号
        A[0].m_key = 1;
        return;
    }
    
    // 构建Huffman树
    A[0].m_key += A[1].m_key; // 合并两个最小频率节点
    root = 0;
    leaf = 2;
    
    // 主循环，构建Huffman树
    for (next=1; next < n-1; next++) {
        // 选择两个最小节点合并
        if (leaf>=n || A[root].m_key<A[leaf].m_key) {
            A[next].m_key = A[root].m_key;
            A[root++].m_key = next;
        } else {
            A[next].m_key = A[leaf++].m_key;
        }
        
        if (leaf>=n || (root<next && A[root].m_key<A[leaf].m_key)) {
            A[next].m_key += A[root].m_key;
            A[root++].m_key = next;
        } else {
            A[next].m_key += A[leaf++].m_key;
        }
    }
    
    // 计算编码长度
    A[n-2].m_key = 0; // 初始化
    for (next=n-3; next>=0; next--) {
        A[next].m_key = A[A[next].m_key].m_key+1; // 计算编码长度
    }
    
    // 调整编码长度以满足前缀编码条件
    avbl = 1;
    used = dpth = 0;
    root = n-2;
    next = n-1;
    
    while (avbl>0) {
        while (root>=0 && (int)A[root].m_key==dpth) {
            used++;
            root--;
        }
        while (avbl>used) {
            A[next--].m_key = dpth;
            avbl--;
        }
        avbl = 2*used; dpth++; used = 0;
    }
}

/**
 * 限制Huffman编码的最大编码长度
 * 确保所有编码长度不超过指定的最大值
 */
static void huffman_enforce_max_code_size(int *pNum_codes, int code_list_len, int max_code_size)
{
    if (code_list_len <= 1) { // 边界情况：编码列表长度小于等于1
        return;
    }

    // 合并超过最大长度的编码
    for (int i = max_code_size + 1; i <= MAX_HUFF_CODESIZE; i++) {
        pNum_codes[max_code_size] += pNum_codes[i];
    }

    // 检查是否满足Kraft不等式
    uint32 total = 0;
    for (int i = max_code_size; i > 0; i--) {
        total += (((uint32)pNum_codes[i]) << (max_code_size - i));
    }

    // 调整编码长度以确保满足Kraft不等式
    while (total != (1UL << max_code_size)) {
        pNum_codes[max_code_size]--; // 减少最大长度的编码数量
        
        // 增加次长编码的数量
        for (int i = max_code_size - 1; i > 0; i--) {
            if (pNum_codes[i]) {
                pNum_codes[i]--;
                pNum_codes[i + 1] += 2;
                break;
            }
        }
        
        total--; // 更新总和
    }
}

/**
 * 哈夫曼表优化函数
 * 基于符号频率生成最优哈夫曼编码表
 * @param table_len 表长度
 */
void huffman_table::optimize(int table_len)
{
    // 初始化符号频率数组
    sym_freq syms0[MAX_HUFF_SYMBOLS], syms1[MAX_HUFF_SYMBOLS];
    
    // 添加特殊的哑元符号(虚拟符号)，确保没有全为1的编码
    // 哈夫曼编码中避免全1编码是为了防止与JPEG标记冲突
    syms0[0].m_key = 1; 
    syms0[0].m_sym_index = 0;
    
    int num_used_syms = 1; // 已使用的符号数(包括哑元)
    
    // 收集所有非零频率的符号
    for (int i = 0; i < table_len; i++) {
        if (m_count[i]) {
            syms0[num_used_syms].m_key = m_count[i]; // 符号频率作为键值
            syms0[num_used_syms++].m_sym_index = i + 1; // 符号索引(加1区分哑元)
        }
    }
    
    // 如果没有符号需要编码，返回
    if (num_used_syms == 1) {
        clear_obj(m_bits);
        clear_obj(m_val);
        m_bits[0] = 1;
        m_val[0] = 0;
        return;
    }
    
    // 按频率排序符号(使用基数排序)
    sym_freq *pSyms = radix_sort_syms(num_used_syms, syms0, syms1);
    
    // 计算最优前缀编码长度
    calculate_minimum_redundancy(pSyms, num_used_syms);
    
    // 统计每个编码长度的符号数量
    int code_sizes[MAX_HUFF_SYMBOLS];
    clear_obj(code_sizes);
    for (int i = 0; i < num_used_syms; i++) {
        code_sizes[pSyms[i].m_sym_index] = pSyms[i].m_key;
    }
    
    // 统计各编码长度的符号数
    clear_obj(m_bits);
    for (int i = 1; i < num_used_syms; i++) {
        m_bits[code_sizes[i]]++;
    }
    
    // 确保编码长度不超过16位(JPEG标准限制)
    huffman_enforce_max_code_size(m_bits, num_used_syms - 1, 16);
    
    // 按编码长度和符号值生成排序的编码表
    int next_code[MAX_HUFF_CODESIZE + 1], code = 0;
    clear_obj(next_code);
    
    // 计算每个编码长度的第一个编码值
    for (int i = 1; i <= 16; i++) {
        next_code[i] = code;
        code = (code + m_bits[i]) << 1;
    }
    
    // 生成符号值数组
    clear_obj(m_val);
    int val_index = 0;
    
    for (int i = 1; i <= 16; i++) {
        for (int j = 0; j < table_len; j++) {
            if (code_sizes[j + 1] == i) {
                m_val[val_index++] = j;
            }
        }
    }
}

/**
 * 发出一个字节
 * 将一个字节写入输出流
 * @param i 要写入的字节
 * @return 是否成功
 */
inline bool jpeg_encoder::emit_byte(uint8 i)
{
    // 检测特殊字节0xFF，需要添加0x00填充
    if (m_all_stream_writes_succeeded) {
        if (m_pOut_buf) {
            *m_pOut_buf++ = i;
            if (--m_out_buf_left == 0) {
                // 输出缓冲区已满，刷新到输出流
                flush_output_buffer();
            }
        } else {
            // 直接写入输出流
            m_all_stream_writes_succeeded = m_pStream->put_buf(&i, 1);
        }
    }
    return m_all_stream_writes_succeeded;
}

/**
 * 发出一个标记
 * 写入JPEG标记(0xFF后跟标记码)
 * @param marker 标记码
 * @return 是否成功
 */
inline bool jpeg_encoder::emit_marker(int marker)
{
    emit_byte(0xFF); // 所有JPEG标记以0xFF开头
    emit_byte((uint8)(marker)); // 标记码
    return m_all_stream_writes_succeeded;
}

/**
 * 发出一个16位字
 * 写入16位整数(高字节在前)
 * @param i 要写入的16位整数
 * @return 是否成功
 */
inline bool jpeg_encoder::emit_word(uint i)
{
    emit_byte((uint8)(i >> 8)); // 高字节
    emit_byte((uint8)(i & 0xFF)); // 低字节
    return m_all_stream_writes_succeeded;
}

/**
 * 计算整数的位数
 * 确定表示一个整数需要的位数
 * @param temp1 输入整数
 * @return 需要的位数
 */
inline uint jpeg_encoder::bit_count(int temp1)
{
    if (temp1 < 0) { // 确保是正数
        temp1 = -temp1;
    }
    
    uint nbits = 0; // 位数计数器
    while (temp1) {
        nbits++; // 递增位计数
        temp1 >>= 1; // 右移一位
    }
    
    return nbits; // 返回位数
}

/**
 * 输出有符号整数位
 * 将有符号整数按指定位数写入位流
 * @param num 有符号整数
 * @param len 位数
 */
inline void jpeg_encoder::put_signed_int_bits(int num, uint len)
{
    // 负数的特殊处理：将负数转为正数表示的形式
    if (num < 0) {
        num--;
        num &= ((1 << len) - 1);
    }
    
    // 输出指定位数的位
    put_bits(num, len);
}

/**
 * 输出位
 * 将指定位数写入位流缓冲
 * @param bits 位值
 * @param len 位数
 */
void jpeg_encoder::put_bits(uint bits, uint len)
{
    m_bit_buffer |= ((uint32)bits << (24 - (m_bits_in += len))); // 将位添加到缓冲
    
    // 当缓冲区至少有8位时，输出字节
    while (m_bits_in >= 8) {
        uint8 c;
        c = (uint8)((m_bit_buffer >> 16) & 0xFF); // 获取高8位
        emit_byte(c); // 输出字节
        
        // 如果输出0xFF，需要添加0x00作为填充(JPEG规范要求)
        if (c == 0xFF) {
            emit_byte(0);
        }
        
        m_bit_buffer <<= 8; // 左移8位
        m_bits_in -= 8; // 减少位计数
    }
}

/**
 * 编码一个8x8块
 * 对一个8x8的量化DCT系数块进行Huffman编码
 * @param src 源量化DCT系数
 * @param huff Huffman表
 * @param comp 组件信息
 * @param write 是否写入数据(false时只统计频率)
 */
void jpeg_encoder::code_block(dctq_t *src, huffman_dcac *huff, component *comp, bool write)
{
    // 编码DC系数(差分编码)
    int dc_delta = src[0] - comp->m_last_dc_val; // 计算与上一个DC值的差值
    comp->m_last_dc_val = src[0]; // 更新最后的DC值
    
    // 处理DC差值
    if (dc_delta == 0) {
        // 差值为0，使用特殊编码
        if (write) {
            put_bits(huff->dc.m_codes[0], huff->dc.m_code_sizes[0]); // 写入0值的哈夫曼编码
        } else {
            huff->dc.m_count_codes[0]++; // 增加0值的计数
            huff->dc.m_count[huff->dc.m_code_sizes[0]]++; // 更新编码长度统计
        }
    } else {
        // 计算差值需要的位数
        uint nbits = bit_count(dc_delta);
        
        // 第一部分：发送位数的哈夫曼编码
        if (write) {
            put_bits(huff->dc.m_codes[nbits], huff->dc.m_code_sizes[nbits]); // 写入位数的哈夫曼编码
            put_signed_int_bits(dc_delta, nbits); // 写入差值本身
        } else {
            huff->dc.m_count_codes[0]++; // 增加符号计数
            huff->dc.m_count[huff->dc.m_code_sizes[nbits]]++; // 更新编码长度统计
        }
    }
    
    // 编码AC系数(使用游程长度编码)
    int run = 0; // 连续0的数量
    
    // 按zigzag顺序处理AC系数
    for (int i = 1; i < 64; i++) {
        const int ac_val = src[zigzag[i]]; // 按zigzag顺序获取系数
        
        if (ac_val == 0) {
            run++; // 累计0的个数
            continue;
        }
        
        // 处理16个以上连续0的情况(ZRL编码)
        while (run >= 16) {
            if (write) {
                put_bits(huff->ac.m_codes[0xF0], huff->ac.m_code_sizes[0xF0]); // 写入ZRL编码(16个0)
            } else {
                huff->ac.m_count_codes[1]++; // 增加ZRL符号计数
                huff->ac.m_count[huff->ac.m_code_sizes[0xF0]]++; // 更新编码长度统计
            }
            run -= 16; // 减去已编码的16个0
        }
        
        // 计算AC系数的位数
        uint nbits = bit_count(ac_val);
        
        // 构造(游程,大小)对符号
        uint symbol = (run << 4) | nbits; // 高4位是游程，低4位是位数
        
        if (write) {
            put_bits(huff->ac.m_codes[symbol], huff->ac.m_code_sizes[symbol]); // 写入符号的哈夫曼编码
            put_signed_int_bits(ac_val, nbits); // 写入系数值
        } else {
            huff->ac.m_count_codes[1]++; // 增加符号计数
            huff->ac.m_count[huff->ac.m_code_sizes[symbol]]++; // 更新编码长度统计
        }
        
        run = 0; // 重置连续0计数
    }
    
    // 如果块以0结尾，发送EOB标记(0运行长度,0位)
    if (run) {
        if (write) {
            put_bits(huff->ac.m_codes[0], huff->ac.m_code_sizes[0]); // 写入EOB编码
        } else {
            huff->ac.m_count_codes[1]++; // 增加EOB符号计数
            huff->ac.m_count[huff->ac.m_code_sizes[0]]++; // 更新编码长度统计
        }
    }
}

/**
 * 编码一行MCU
 * 处理图像中一行的最小编码单元(MCU)
 * @param y 行的Y坐标
 * @param write 是否写入数据(false时只统计频率)
 */
void jpeg_encoder::code_mcu_row(int y, bool write)
{
    if (m_num_components == 1) { // 灰度图像
        for (int x = 0; x < m_x; x += m_mcu_w) {
            code_block(m_image[0].get_dctq(x, y), &m_huff[0], &m_comp[0], write); // 只编码Y分量
        }
    } else if ((m_comp[0].m_h_samp == 1) && (m_comp[0].m_v_samp == 1)) { // 无子采样(1x1)
        for (int x = 0; x < m_x; x += m_mcu_w) {
            code_block(m_image[0].get_dctq(x, y), &m_huff[0], &m_comp[0], write); // Y分量
            code_block(m_image[1].get_dctq(x, y), &m_huff[1], &m_comp[1], write); // Cb分量
            code_block(m_image[2].get_dctq(x, y), &m_huff[1], &m_comp[2], write); // Cr分量
        }
    } else if ((m_comp[0].m_h_samp == 2) && (m_comp[0].m_v_samp == 1)) { // 水平2:1子采样
        for (int x = 0; x < m_x; x += m_mcu_w) {
            code_block(m_image[0].get_dctq(x,   y), &m_huff[0], &m_comp[0], write); // Y分量左侧块
            code_block(m_image[0].get_dctq(x+8, y), &m_huff[0], &m_comp[0], write); // Y分量右侧块
            code_block(m_image[1].get_dctq(x/2, y), &m_huff[1], &m_comp[1], write); // Cb分量(子采样)
            code_block(m_image[2].get_dctq(x/2, y), &m_huff[1], &m_comp[2], write); // Cr分量(子采样)
        }
    } else if ((m_comp[0].m_h_samp == 2) && (m_comp[0].m_v_samp == 2)) { // 水平和垂直2:1子采样
        for (int x = 0; x < m_x; x += m_mcu_w) {
            code_block(m_image[0].get_dctq(x,   y),   &m_huff[0], &m_comp[0], write); // Y分量左上块
            code_block(m_image[0].get_dctq(x+8, y),   &m_huff[0], &m_comp[0], write); // Y分量右上块
            code_block(m_image[0].get_dctq(x,   y+8), &m_huff[0], &m_comp[0], write); // Y分量左下块
            code_block(m_image[0].get_dctq(x+8, y+8), &m_huff[0], &m_comp[0], write); // Y分量右下块
            code_block(m_image[1].get_dctq(x/2, y/2), &m_huff[1], &m_comp[1], write); // Cb分量(子采样)
            code_block(m_image[2].get_dctq(x/2, y/2), &m_huff[1], &m_comp[2], write); // Cr分量(子采样)
        }
    }
}

/**
 * 输出结束标记
 * 完成位流并写入JPEG的结束标记
 * @return 写入是否成功
 */
bool jpeg_encoder::emit_end_markers()
{
    put_bits(0x7F, 7); // 填充位以确保字节对齐
    flush_output_buffer(); // 刷新输出缓冲区
    emit_marker(M_EOI); // 图像结束标记
    return m_all_stream_writes_succeeded; // 返回是否所有写入都成功
}

/**
 * 压缩图像
 * JPEG压缩的主要函数，处理整个压缩流程
 * @return 压缩是否成功
 */
bool jpeg_encoder::compress_image()
{
    // 第1阶段：对每个8x8块进行DCT和量化
    for(int c=0; c < m_num_components; c++) { // 处理每个颜色分量
        for (int y = 0; y < m_image[c].m_y; y+= 8) { // 垂直方向每8像素一块
            for (int x = 0; x < m_image[c].m_x; x += 8) { // 水平方向每8像素一块
                dct_t sample[64]; // 临时DCT计算缓冲区
                m_image[c].load_block(sample, x, y); // 加载8x8像素块
                quantize_pixels(sample, m_image[c].get_dctq(x, y), m_huff[c > 0].m_quantization_table); // 应用DCT和量化
            }
        }
    }

    // 第2阶段：统计分析，计算Huffman表的符号频率
    for (int y = 0; y < m_y; y+= m_mcu_h) { // 处理每行MCU
        code_mcu_row(y, false); // 只统计频率，不实际写入
    }
    
    // 第3阶段：基于统计优化Huffman表
    compute_huffman_tables();
    
    // 第4阶段：重置DC预测值
    reset_last_dc();

    // 第5阶段：写入文件头和标记
    emit_start_markers();
    
    // 第6阶段：实际编码并写入压缩数据
    for (int y = 0; y < m_y; y+= m_mcu_h) {
        if (!m_all_stream_writes_succeeded) { // 检查写入错误
            return false;
        }
        code_mcu_row(y, true); // 编码并写入数据
    }
    
    // 第7阶段：完成文件
    return emit_end_markers();
}

/**
 * 加载MCU的Y分量
 * 加载灰度数据到图像Y分量
 * @param pSrc 源数据
 * @param width 宽度
 * @param bpp 每像素字节数
 * @param y 行索引
 */
void jpeg_encoder::load_mcu_Y(const uint8 *pSrc, int width, int bpp, int y)
{
    if (bpp == 4) { // RGBA格式
        RGB_to_Y(m_image[0], reinterpret_cast<const rgba *>(pSrc), width, y);
    } else if (bpp == 3) { // RGB格式
        RGB_to_Y_AVX(m_image[0], reinterpret_cast<const rgb *>(pSrc), width, y);
    } else // 灰度格式
        for(int x=0; x < width; x++) {
            m_image[0].set_px(pSrc[x]-128.0, x, y); // 转换为有符号范围
        }

    // 复制图像边缘像素以填充到MCU边界
    const float lastpx = m_image[0].get_px(width - 1, y);
    for (int x = width; x < m_image[0].m_x; x++) {
        m_image[0].set_px(lastpx, x, y); // 复制最后一个像素
    }
}

/**
 * 加载MCU的YCC分量
 * 加载彩色数据到图像的YCbCr分量
 * @param pSrc 源数据
 * @param width 宽度
 * @param bpp 每像素字节数
 * @param y 行索引
 */
void jpeg_encoder::load_mcu_YCC(const uint8 *pSrc, int width, int bpp, int y)
{
    if (bpp == 4) { // RGBA格式
        RGB_to_YCC_AVX2(m_image, reinterpret_cast<const rgba *>(pSrc), width, y);
    } else if (bpp == 3) { // RGB格式
        RGB_to_YCC_AVX2(m_image, reinterpret_cast<const rgb *>(pSrc), width, y);
    } else { // 灰度格式
        Y_to_YCC(m_image, pSrc, width, y);
    }

    // 复制图像边缘像素以填充到MCU边界
    for(int c=0; c < m_num_components; c++) {
        const float lastpx = m_image[c].get_px(width - 1, y);
        for (int x = width; x < m_image[0].m_x; x++) {
            m_image[c].set_px(lastpx, x, y); // 复制最后一个像素
        }
    }
}

/**
 * 清除编码器状态
 * 重置编码器到初始状态
 */
void jpeg_encoder::clear()
{
    m_num_components=0; // 重置组件数量
    m_all_stream_writes_succeeded = true; // 重置写入状态
}

/**
 * 构造函数
 * 初始化JPEG编码器
 */
jpeg_encoder::jpeg_encoder()
{
    clear(); // 初始化到干净状态
}

/**
 * 析构函数
 * 清理JPEG编码器资源
 */
jpeg_encoder::~jpeg_encoder()
{
    deinit(); // 释放资源
}

/**
 * 初始化函数
 * 设置编码器参数并准备编码
 * @param pStream 输出流
 * @param width 图像宽度
 * @param height 图像高度
 * @param comp_params 压缩参数
 * @return 初始化是否成功
 */
bool jpeg_encoder::init(output_stream *pStream, int width, int height, const params &comp_params)
{
    deinit(); // 先清理之前的资源
    if (!pStream || width < 1 || height < 1 || !comp_params.check()) { // 参数验证
        return false;
    }
    m_pStream = pStream; // 设置输出流
    m_params = comp_params; // 设置压缩参数
    return jpg_open(width, height); // 打开编码器
}

/**
 * 释放资源函数
 * 清理编码器使用的所有资源
 */
void jpeg_encoder::deinit()
{
    for(int c=0; c < m_num_components; c++) {
        m_image[c].deinit(); // 释放图像分量的内存
    }
    clear(); // 重置状态
}

/**
 * 读取图像数据函数
 * 从内存缓冲区读取图像数据并准备压缩
 * @param image_data 源图像数据
 * @param width 图像宽度
 * @param height 图像高度
 * @param bpp 每像素字节数(1=灰度，3=RGB，4=RGBA)
 * @return 读取是否成功
 */
bool jpeg_encoder::read_image(const uint8 *image_data, int width, int height, int bpp)
{
    if (bpp != 1 && bpp != 3 && bpp != 4) { // 验证颜色格式
        return false;
    }

    // 加载图像数据到内部缓冲区
    for (int y = 0; y < height; y++) {
        if (m_num_components == 1) { // 灰度图像
            load_mcu_Y(image_data + width * y * bpp, width, bpp, y);
        } else { // 彩色图像
            load_mcu_YCC(image_data + width * y * bpp, width, bpp, y);
        }
    }

    // 复制最后一行到边缘，以填充到MCU边界
    for(int c=0; c < m_num_components; c++) {
        for (int y = height; y < m_image[c].m_y; y++) {
            for(int x=0; x < m_image[c].m_x; x++) {
                m_image[c].set_px(m_image[c].get_px(x, y-1), x, y);
            }
        }
    }

    // 对色度分量进行子采样(如果需要)
    if (m_comp[0].m_h_samp == 2) {
        for(int c=1; c < m_num_components; c++) {
            m_image[c].subsample(m_image[0], m_comp[0].m_v_samp);
        }
    }

    // 溢出白色和黑色，使失真也溢出，以便解码器截断失真(如振铃效应)
    if (m_huff[0].m_quantization_table[0] > 2) {
        for(int c=0; c < m_num_components; c++) {
            for(int y=0; y < m_image[c].m_y; y++) {
                for(int x=0; x < m_image[c].m_x; x++) {
                    float px = m_image[c].get_px(x,y);
                    if (px <= -128.f) {
                        px -= m_huff[0].m_quantization_table[0]; // 黑色溢出
                    } else if (px >= 128.f) {
                        px += m_huff[0].m_quantization_table[0]; // 白色溢出
                    }
                    m_image[c].set_px(px, x, y);
                }
            }
        }
    }

    return true;
}

/**
 * 加载一个8x8像素块
 * 从图像中加载一个8x8像素块到DCT处理缓冲区
 * @param pDst 目标缓冲区
 * @param x 块的X坐标
 * @param y 块的Y坐标
 */
void image::load_block(dct_t *pDst, int x, int y) 
{
    // 每行的处理(一个8x8块有8行)
    for (int r = 0; r < 8; r++) {
        // 确保Y坐标在图像范围内
        const int by = JPGE_MIN(JPGE_MAX(y + r, 0), m_y - 1);
        
        // 当前行的起始像素位置
        const float *pSrc = &m_pixels[by * m_x];
        
        // 每列的处理(一个8x8块有8列)
        for (int c = 0; c < 8; c++) {
            // 确保X坐标在图像范围内
            const int bx = JPGE_MIN(JPGE_MAX(x + c, 0), m_x - 1);
            
            // 将像素值复制到目标缓冲区(转换为DCT输入格式)
            pDst[c + r * 8] = pSrc[bx];
        }
    }
}

/**
 * 2x1像素混合
 * 计算两个水平相邻像素的加权平均，用于水平色度子采样
 * @param x 起始X坐标
 * @param y Y坐标
 * @param luma 亮度图像(用于权重计算)
 * @return 混合后的像素值
 */
float image::blend_dual(int x, int y, image &luma) 
{
    // 获取对应亮度图像中的两个像素位置，并确保亮度值为正(用于权重计算)
    float y00 = JPGE_MAX(0, luma.get_px(x*2,   y) + 128.0);
    float y01 = JPGE_MAX(0, luma.get_px(x*2+1, y) + 128.0);
    
    // 计算亮度和，用于权重
    float l = y00 + y01;
    
    if (!l) { // 避免除以0
        return 0;
    }
    
    // 获取当前图像(色度)中对应的两个像素
    float v00 = get_px(x*2,   y);
    float v01 = get_px(x*2+1, y);
    
    // 基于亮度权重计算色度平均值
    return (v00 * y00 + v01 * y01) / l;
}

/**
 * 2x2像素混合
 * 计算四个相邻像素的加权平均，用于水平和垂直色度子采样
 * @param x 起始X坐标
 * @param y 起始Y坐标
 * @param luma 亮度图像(用于权重计算)
 * @return 混合后的像素值
 */
float image::blend_quad(int x, int y, image &luma) 
{
    // 获取对应亮度图像中的4个像素位置，并确保亮度值为正(用于权重计算)
    float y00 = JPGE_MAX(0, luma.get_px(x*2,   y*2)   + 128.0);
    float y01 = JPGE_MAX(0, luma.get_px(x*2+1, y*2)   + 128.0);
    float y10 = JPGE_MAX(0, luma.get_px(x*2,   y*2+1) + 128.0);
    float y11 = JPGE_MAX(0, luma.get_px(x*2+1, y*2+1) + 128.0);
    
    // 计算亮度和，用于权重
    float l = y00 + y01 + y10 + y11;
    
    if (!l) { // 避免除以0
        return 0;
    }
    
    // 获取当前图像(色度)中对应的四个像素
    float v00 = get_px(x*2,   y*2);
    float v01 = get_px(x*2+1, y*2);
    float v10 = get_px(x*2,   y*2+1);
    float v11 = get_px(x*2+1, y*2+1);
    
    // 基于亮度权重计算色度平均值
    return (v00 * y00 + v01 * y01 + v10 * y10 + v11 * y11) / l;
}

/**
 * 量化至零函数
 * 对DCT系数进行量化，向零方向舍入
 * @param j DCT系数
 * @param quant 量化值
 * @return 量化后的系数
 */
inline dctq_t round_to_zero(const dct_t j, const int32 quant)
{
    if (j < 0) {
        // 处理负值：加上半个量化步长然后向下取整
        dct_t jtmp = -j + (quant >> 1);
        return (jtmp < quant) ? 0 : static_cast<dctq_t>(-(jtmp / quant));
    } else {
        // 处理正值：加上半个量化步长然后向下取整
        dct_t jtmp = j + (quant >> 1);
        return (jtmp < quant) ? 0 : static_cast<dctq_t>(jtmp / quant);
    }
}

/**
 * 打开JPEG编码器
 * 初始化编码器参数和分配内存
 * @param width 图像宽度
 * @param height 图像高度
 * @return 初始化是否成功
 */
bool jpeg_encoder::jpg_open(int width, int height)
{
    // 基于子采样方式设置编码器参数
    switch (m_params.m_subsampling) {
    case Y_ONLY: // 只有亮度(灰度图像)
        m_num_components = 1;
        m_comp[0].m_h_samp = 1; m_comp[0].m_v_samp = 1;
        m_mcu_w = 8; m_mcu_h = 8;
        break;
    case H1V1: // 无子采样
        m_num_components = 3;
        m_comp[0].m_h_samp = 1; m_comp[0].m_v_samp = 1;
        m_comp[1].m_h_samp = 1; m_comp[1].m_v_samp = 1;
        m_comp[2].m_h_samp = 1; m_comp[2].m_v_samp = 1;
        m_mcu_w = 8; m_mcu_h = 8;
        break;
    case H2V1: // 水平2:1子采样
        m_num_components = 3;
        m_comp[0].m_h_samp = 2; m_comp[0].m_v_samp = 1;
        m_comp[1].m_h_samp = 1; m_comp[1].m_v_samp = 1;
        m_comp[2].m_h_samp = 1; m_comp[2].m_v_samp = 1;
        m_mcu_w = 16; m_mcu_h = 8;
        break;
    default:
    case H2V2: // 水平和垂直都是2:1子采样
        m_num_components = 3;
        m_comp[0].m_h_samp = 2; m_comp[0].m_v_samp = 2;
        m_comp[1].m_h_samp = 1; m_comp[1].m_v_samp = 1;
        m_comp[2].m_h_samp = 1; m_comp[2].m_v_samp = 1;
        m_mcu_w = 16; m_mcu_h = 16;
    }

    // 保存图像尺寸
    m_x = width;
    m_y = height;

    // 计算每个分量的尺寸(向上取整到MCU边界)
    m_image[0].m_x = (m_x + m_mcu_w - 1) & (~(m_mcu_w - 1));
    m_image[0].m_y = (m_y + m_mcu_h - 1) & (~(m_mcu_h - 1));
    
    m_image[1].m_x = m_image[0].m_x >> (m_comp[0].m_h_samp - m_comp[1].m_h_samp);
    m_image[1].m_y = m_image[0].m_y >> (m_comp[0].m_v_samp - m_comp[1].m_v_samp);
    
    m_image[2].m_x = m_image[1].m_x;
    m_image[2].m_y = m_image[1].m_y;

    // 为每个分量分配内存
    for (int c = 0; c < m_num_components; c++) {
        if (!m_image[c].init()) {
            return false;
        }
    }

    // 计算量化表，标准亮度和色度表会基于质量参数进行缩放
    compute_quant_table(m_huff[0].m_quantization_table, s_std_lum_quant);
    compute_quant_table(m_huff[1].m_quantization_table, m_params.m_no_chroma_discrim_flag ? s_std_lum_quant : s_std_croma_quant);

    // 初始化输出缓冲区
    m_out_buf_left = JPGE_OUT_BUF_SIZE;
    m_pOut_buf = m_out_buf;

    return true;
}

} // namespace jpge
