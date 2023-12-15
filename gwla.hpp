/* BSD 3-Clause License

Copyright (c) 2023, mahdi kashani

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef GWLA_HPP
#define GWLA_HPP

#include <assert.h>
#include <cmath>
#include <cstdint>
#include <iostream>


#define min(a, b) (a <= b ? a : b)
#define max(a, b) (a >= b ? a : b)

namespace GW {

// TODO: Write test
// TODO: Type checking is too strict


template <typename T, size_t N> class __Vec_impl;
template <typename T> class __Vec2_impl;
template <typename T> class __Vec3_impl;
template <typename T> class __Vec4_impl;


template <typename T, size_t N, size_t M> class __Matrix_impl;
template <typename T> class __Matrix2_impl;
template <typename T> class __Matrix3_impl;
template <typename T> class __Matrix4_impl;


template<size_t N> using Vec_f = __Vec_impl<float, N>;
template<size_t N> using Vec_d = __Vec_impl<double, N>;
#define Vec Vec_f

template<size_t N> using Vec_i64 = __Vec_impl<int64_t, N>;
template<size_t N> using Vec_i32 = __Vec_impl<int32_t, N>;
template<size_t N> using Vec_i16 = __Vec_impl<int16_t, N>;
template<size_t N> using Vec_i8 = __Vec_impl<int8_t, N>;
#define Vec_i Vec_i32

template<size_t N> using Vec_u64 = __Vec_impl<uint64_t, N>;
template<size_t N> using Vec_u32 = __Vec_impl<uint32_t, N>;
template<size_t N> using Vec_u16 = __Vec_impl<uint16_t, N>;
template<size_t N> using Vec_u8 = __Vec_impl<uint8_t, N>;


typedef __Vec2_impl<float>       Vec2;
typedef __Vec2_impl<int8_t>      Vec2_i8;
typedef __Vec2_impl<int16_t>     Vec2_i16;
typedef __Vec2_impl<int32_t>     Vec2_i32;
typedef __Vec2_impl<uint8_t>     Vec2_u8;
typedef __Vec2_impl<uint16_t>    Vec2_u16;
typedef __Vec2_impl<uint32_t>    Vec2_u32;
typedef __Vec2_impl<double>      Vec2_d;


typedef __Vec3_impl<float>       Vec3;
typedef __Vec3_impl<int8_t>      Vec3_i8;
typedef __Vec3_impl<int16_t>     Vec3_i16;
typedef __Vec3_impl<int32_t>     Vec3_i32;
typedef __Vec3_impl<uint8_t>     Vec3_u8;
typedef __Vec3_impl<uint16_t>    Vec3_u16;
typedef __Vec3_impl<uint32_t>    Vec3_u32;
typedef __Vec3_impl<double>      Vec3_d;


typedef __Vec4_impl<float>       Vec4;
typedef __Vec4_impl<int8_t>      Vec4_i8;
typedef __Vec4_impl<int16_t>     Vec4_i16;
typedef __Vec4_impl<int32_t>     Vec4_i32;
typedef __Vec4_impl<uint8_t>     Vec4_u8;
typedef __Vec4_impl<uint16_t>    Vec4_u16;
typedef __Vec4_impl<uint32_t>    Vec4_u32;
typedef __Vec4_impl<double>      Vec4_d;


template<size_t N, size_t  M> using Mat_f = __Matrix_impl<float, N, M>;
template<size_t N, size_t  M> using Mat_d = __Matrix_impl<double, N, M>;
#define Mat Mat_f

template<size_t N, size_t  M> using Mat_i64 = __Matrix_impl<int64_t, N, M>;
template<size_t N, size_t  M> using Mat_i32 = __Matrix_impl<int32_t, N, M>;
template<size_t N, size_t  M> using Mat_i16 = __Matrix_impl<int16_t, N, M>;
template<size_t N, size_t  M> using Mat_i8 = __Matrix_impl<int8_t, N, M>;
#define Mat_i Mat_i32

template<size_t N, size_t  M> using Mat_u64 = __Matrix_impl<uint64_t, N, M>;
template<size_t N, size_t  M> using Mat_u32 = __Matrix_impl<uint32_t, N, M>;
template<size_t N, size_t  M> using Mat_u16 = __Matrix_impl<uint16_t, N, M>;
template<size_t N, size_t  M> using Mat_u8 = __Matrix_impl<uint8_t, N, M>;


typedef __Matrix2_impl<float>       Mat2;
typedef __Matrix2_impl<int8_t>      Mat2_i8;
typedef __Matrix2_impl<int16_t>     Mat2_i16;
typedef __Matrix2_impl<int32_t>     Mat2_i32;
typedef __Matrix2_impl<uint8_t>     Mat2_u8;
typedef __Matrix2_impl<uint16_t>    Mat2_u16;
typedef __Matrix2_impl<uint32_t>    Mat2_u32;
typedef __Matrix2_impl<double>      Mat2_d;


typedef __Matrix3_impl<float>       Mat3;
typedef __Matrix3_impl<int8_t>      Mat3_i8;
typedef __Matrix3_impl<int16_t>     Mat3_i16;
typedef __Matrix3_impl<int32_t>     Mat3_i32;
typedef __Matrix3_impl<uint8_t>     Mat3_u8;
typedef __Matrix3_impl<uint16_t>    Mat3_u16;
typedef __Matrix3_impl<uint32_t>    Mat3_u32;
typedef __Matrix3_impl<double>      Mat3_d;


typedef __Matrix4_impl<float>       Mat4;
typedef __Matrix4_impl<int8_t>      Mat4_i8;
typedef __Matrix4_impl<int16_t>     Mat4_i16;
typedef __Matrix4_impl<int32_t>     Mat4_i32;
typedef __Matrix4_impl<uint8_t>     Mat4_u8;
typedef __Matrix4_impl<uint16_t>    Mat4_u16;
typedef __Matrix4_impl<uint32_t>    Mat4_u32;
typedef __Matrix4_impl<double>      Mat4_d;


template <typename T, size_t N>
class __Vec_impl {
protected:
    static const size_t m_size = N;
    T * m_data;

public:
    __Vec_impl() { m_data = reinterpret_cast<T*>( new T[m_size]() ); };
    __Vec_impl(T value) {
        m_data = new T[m_size];
        std::fill(m_data, m_data + m_size, value);
    }
    __Vec_impl(T *data_) { m_data = data_; }
    __Vec_impl(__Vec_impl<T, N> &other) {
        m_data = reinterpret_cast<T*>( new T[other.size()] {0} );
        std::copy(other.m_data, other.m_data + other.size(), m_data);
    }

    template<typename... Args,
             std::enable_if_t<std::is_convertible<std::common_type_t<Args...>, T>::value, bool> = true>
    __Vec_impl(Args... values)
    {
        using Args_T = std::common_type_t<Args...>;
        static_assert(N == sizeof ...(values), "Your input m_data m_size is not match with vector m_size");
        m_data = reinterpret_cast<T*>( new Args_T[m_size] {0} );
        size_t i = 0;
        for (Args_T val : {values...}) {
            m_data[i++] = static_cast<T>(val);
        }
    }

    ~__Vec_impl() { delete[] m_data; }

    T& operator() (size_t i) {
        assert(i < size() && "Index is out of range");
        return m_data[i];
    }
    const T& operator() (size_t i) const {
        assert(i < size() && "Index is out of range");
        return m_data[i];
    }

    T& operator[] (size_t i) {
        assert(i < size() && "Index is out of range");
        return m_data[i];
    }
    const T& operator[] (size_t i) const {
        assert(i < size() && "Index is out of range");
        return m_data[i];
    }

    __Vec_impl& operator=(const __Vec_impl& other) {
        assert(this->size() == other.size());
        if (this == &other)
            return *this;
        for (size_t i = 0 ; i < size() ; i++) {
            std::copy(other.m_data,
                      other.m_data + other.size(),
                      this->m_data);
        }
        return *this;
    }

    friend __Vec_impl& operator+=(__Vec_impl& lhs, const __Vec_impl& rhs) {
        assert(lhs.size() == rhs.size());
        for (size_t i = 0 ; i < size() ; i++)
            lhs(i) = lhs(i) + rhs(i);
        return lhs;
    }

    friend __Vec_impl& operator+(const __Vec_impl& lhs,
                                 const __Vec_impl& rhs) {
        assert(lhs.size() == rhs.size());
        __Vec_impl *vec = new __Vec_impl;
        for (size_t i = 0 ; i < lhs.size() ; i++)
            (*vec)(i) = lhs(i) + rhs(i);
        return *vec;
    }

    friend __Vec_impl& operator+=(__Vec_impl& lhs, const T& value) {
        for (size_t i = 0 ; i < lhs.size() ; i++)
            lhs(i) = lhs(i) + value;
        return lhs;
    }

    friend __Vec_impl& operator+(const __Vec_impl& lhs, const T& value) {
        __Vec_impl *vec = new __Vec_impl;
        for (size_t i = 0 ; i < lhs.size() ; i++)
            (*vec)(i) = lhs(i) + value;
        return *vec;
    }

    friend __Vec_impl& operator-=(__Vec_impl& lhs, const __Vec_impl& rhs) {
        assert(lhs.size() == rhs.size());
        for (size_t i = 0 ; i < lhs.size() ; i++)
            lhs(i) = lhs(i) - rhs(i);
        return lhs;
    }

    friend __Vec_impl& operator-(const __Vec_impl& lhs,
                                 const __Vec_impl& rhs) {
        assert(lhs.size() == rhs.size());
        __Vec_impl *vec = new __Vec_impl;
        for (size_t i = 0 ; i < lhs.size() ; i++)
            (*vec)(i) = lhs(i) - rhs(i);
        return *vec;
    }

    friend __Vec_impl& operator-=(__Vec_impl& lhs, const T& value) {
        for (size_t i = 0 ; i < lhs.size() ; i++)
            lhs(i) = lhs(i) - value;
        return lhs;
    }

    friend __Vec_impl& operator-(const __Vec_impl& lhs, const T& value) {
        __Vec_impl *vec = new __Vec_impl;
        for (size_t i = 0 ; i < lhs.size() ; i++)
            (*vec)(i) = lhs(i) - value;
        return *vec;
    }

    friend __Vec_impl& operator*=(__Vec_impl& lhs, const __Vec_impl& rhs) {
        assert(lhs.size() == rhs.size());
        for (size_t i = 0 ; i < lhs.size() ; i++)
            lhs(i) = lhs(i) * rhs(i);
        return lhs;
    }

    friend __Vec_impl& operator*(const __Vec_impl& lhs,
                                 const __Vec_impl& rhs) {
        assert(lhs.size() == rhs.size());
        __Vec_impl *vec = new __Vec_impl;
        for (size_t i = 0 ; i < lhs.size() ; i++)
            (*vec)(i) = lhs(i) * rhs(i);
        return *vec;
    }

    friend __Vec_impl& operator*=(__Vec_impl& lhs, const T& value) {
        for (size_t i = 0 ; i < lhs.size() ; i++)
            lhs(i) = lhs(i) * value;
        return lhs;
    }

    friend __Vec_impl& operator*(const __Vec_impl& lhs, const T& value) {
        __Vec_impl *vec = new __Vec_impl;
        for (size_t i = 0 ; i < lhs.size() ; i++)
            (*vec)(i) = lhs(i) * value;
        return *vec;
    }

    friend __Vec_impl& operator/=(__Vec_impl& lhs, const __Vec_impl& rhs) {
        assert(lhs.size() == rhs.size());
        for (size_t i = 0 ; i < lhs.size() ; i++)
            lhs(i) = lhs(i) - rhs(i);
        return lhs;
    }

    friend __Vec_impl& operator/(const __Vec_impl& lhs,
                                 const __Vec_impl& rhs) {
        assert(lhs.size() == rhs.size());
        __Vec_impl *vec = new __Vec_impl;
        for (size_t i = 0 ; i < lhs.size() ; i++)
            (*vec)(i) = lhs(i) / rhs(i);
        return *vec;
    }

    friend __Vec_impl& operator/=(__Vec_impl& lhs, const T& value) {
        for (size_t i = 0 ; i < lhs.size() ; i++)
            lhs(i) = lhs(i) / value;
        return lhs;
    }

    friend __Vec_impl& operator/(const __Vec_impl& lhs, const T& value) {
        __Vec_impl *vec = new __Vec_impl;
        for (size_t i = 0 ; i < lhs.size() ; i++)
            (*vec)(i) = lhs(i) / value;
        return *vec;
    }

    friend __Vec_impl& operator%=(__Vec_impl& lhs, const T& value) {
        for (size_t i = 0 ; i < lhs.size() ; i++)
            lhs(i) = lhs(i) % value;
        return lhs;
    }

    friend __Vec_impl& operator%(const __Vec_impl& lhs, const T& value) {
        __Vec_impl *vec = new __Vec_impl;
        for (size_t i = 0 ; i < lhs.size() ; i++)
            (*vec)(i) = lhs(i) % value;
        return *vec;
    }

    friend __Vec_impl& operator-(const __Vec_impl& vec) {
        __Vec_impl *neg_vec = new __Vec_impl;
        for (size_t i = 0 ; i < vec.size() ; i++)
            (*neg_vec)(i) = -1 * vec(i);
        return *neg_vec;
    }

    T dot(const __Vec_impl& other) {
        assert(this->size() == other.size());
        T res {0};
        for (size_t i = 0 ; i < this->size() ; i++)
            res += m_data[i] * other(i);
        return res;
    }

    friend T& operator&(const __Vec_impl& lhs, const __Vec_impl& rhs) {
        T *res = new T;
        *res = 0;
        for (size_t i = 0 ; i < lhs.size() ; i++) {
            *res += lhs(i) * rhs(i);
        }
        return *res;
    }

   friend std::ostream& operator<<(std::ostream& os, __Vec_impl<T, N> const &vec) {
        std::string str = "<";
        for (size_t i = 0; i < vec.size(); i++) {
            str += std::to_string(vec(i));
            if (i != vec.size() - 1) str += ' ';
        }
        str += ">";
        return os << str;
   }

    __Vec_impl<double, N>& pow(const double power) const {
        __Vec_impl<double, N> *res = new __Vec_impl<double, N>;
        for (size_t i = 0 ; i < size() ; i++) {
            res[i] = std::pow(static_cast<double>(m_data[i]), power);
        }
        return *res;
    }

    float vec_lenght() {
        float lenght = 0;
        for (int i = 0 ; i < size() ; i++) {
            lenght += (m_data[i] * m_data[i]);
        }
        return sqrt(lenght);
    }

    void normalize() {
        float lenght = vec_lenght();
        (*this) /= lenght;
    }

    static __Vec_impl<float, N> normalize(__Vec_impl<float, N> vec) {
        __Vec_impl<float, N> *norm_vec = new __Vec_impl<float, N> {0.0f};
        *norm_vec = vec;
        norm_vec.normilize();
        return *norm_vec;
    }

    virtual inline const size_t size() const { return m_size; }
};


template <typename T>
class __Vec2_impl : public __Vec_impl<T, 2> {
    using __Vec_impl<T, 2>::m_data;

public:
    T& x = m_data[0];
    T& y = m_data[1];

    using __Vec_impl<T, 2>::__Vec_impl;
};


template <typename T>
class __Vec3_impl : public __Vec_impl<T, 3> {
    using __Vec_impl<T, 3>::m_data;

public:
    T& x = m_data[0];
    T& y = m_data[1];
    T& z = m_data[2];

    using __Vec_impl<T, 3>::__Vec_impl;

    __Vec3_impl& operator=(const __Vec_impl<T, 3>& other) {
        if (this == &other)
            return *this;
        for (size_t i = 0 ; i < 3 ; i++) {
            x = other(0);
            y = other(1);
            z = other(2);
        }
        return *this;
    }

    __Vec3_impl& operator=(__Vec3_impl& other) {
        if (this == &other)
            return *this;
        for (size_t i = 0 ; i < 3 ; i++) {
            std::copy(other.m_data, other.m_data + 3, this->m_data);
        }
        return *this;
    }

    template <typename S, typename R>
    static __Vec3_impl<typename std::common_type<S, R>::type> cross(const __Vec3_impl<S>& x, const __Vec3_impl<R>& y) {
        __Vec3_impl<typename std::common_type<S, R>::type> *cross_vec = new __Vec3_impl<typename std::common_type<S, R>::type>;
        (*cross_vec)(0) = (x(1) * y(2)) - (y(1) * x(2));
        (*cross_vec)(1) = (x(2) * y(0)) - (y(2) * x(0));
        (*cross_vec)(2) = (x(0) * y(1)) - (y(0) * x(1));
        return *cross_vec;
    }


    template <typename S>
    __Vec3_impl<typename std::common_type<T, S>::type> cross(const __Vec3_impl<S>& other) const {
        __Vec3_impl<typename std::common_type<T, S>::type>* cross_vec
            = new __Vec3_impl<typename std::common_type<T, S>::type>;
        (*cross_vec)(0) = ((*this)(1) * other(2)) - (other(1) * (*this)(2));
        (*cross_vec)(1) = ((*this)(2) * other(0)) - (other(2) * (*this)(0));
        (*cross_vec)(2) = ((*this)(0) * other(1)) - (other(0) * (*this)(1));
        return *cross_vec;
    }


    static __Matrix4_impl<T>
    look_at(const __Vec3_impl<T>& eye, const __Vec3_impl<T>& center, const __Vec3_impl<T>& up) {
        __Matrix4_impl<T> matrix;
        __Vec3_impl<T> X, Y, Z;
        Z = eye - center;
        Z.normalize();
        Y = up;
        X = Y.cross(Z);
        Y = Z.cross(X);
        X.normalize();
        Y.normalize();

        matrix(0, 0) = X.x;
        matrix(1, 0) = X.y;
        matrix(2, 0) = X.z;
        matrix(3, 0) = -X.dot(eye);
        matrix(0, 1) = Y.x;
        matrix(1, 1) = Y.y;
        matrix(2, 1) = Y.z;
        matrix(3, 1) = -Y.dot(eye);
        matrix(0, 2) = Z.x;
        matrix(1, 2) = Z.y;
        matrix(2, 2) = Z.z;
        matrix(3, 2) = -Z.dot(eye);
        matrix(0, 3) = 0;
        matrix(1, 3) = 0;
        matrix(2, 3) = 0;
        matrix(3, 3) = 1.0f;

        return matrix;
    }

};


template <typename T>
class __Vec4_impl : public __Vec_impl<T, 4> {
    using __Vec_impl<T, 4>::m_data;

public:
    T& x = m_data[0];
    T& y = m_data[1];
    T& z = m_data[2];
    T& w = m_data[3];

    using __Vec_impl<T, 4>::__Vec_impl;
};


struct MatShape {
    size_t row, col;

    friend bool operator== (const MatShape &lhs, const MatShape &rhs) {
        return ((lhs.col == rhs.col) && (lhs.row == rhs.row));
    }
};

template <typename T, size_t N, size_t M>
class __Matrix_impl {
protected:
    T *m_data;
    static const size_t ROW = N;
    static const size_t COL = M;

public:
    __Matrix_impl() { m_data = reinterpret_cast<T*>( new T[this->size()]() ); }
    explicit __Matrix_impl(T value) {
        m_data = new T[this->size()] {value};
        std::fill(m_data, m_data + size(), value);
    }
    __Matrix_impl(T *data_) { m_data = data_; }
    __Matrix_impl(__Matrix_impl<T, N, M> &other) {
        m_data = reinterpret_cast<T*>( new T[this->size()] {0} );
        std::copy(other.m_data, other.m_data + other.size(), m_data);
    }

    template<typename... Args,
             std::enable_if_t<std::is_convertible<std::common_type_t<Args...>, T>::value, bool> = true>
    __Matrix_impl(Args... values) {
        using Args_T = std::common_type_t<Args...>;
        static_assert(M * N == sizeof ...(values), "Your input m_data size is not match with matrix size");
        m_data = reinterpret_cast<T*>( new Args_T[this->size()] {0} );
        int i = 0;
        for (Args_T val : {values...}) {
            m_data[i++] = static_cast<T>(val);
        }
    }

    ~__Matrix_impl() { delete[] m_data; }

    virtual const MatShape &get_shape() const {
        static MatShape SHAPE {ROW, COL};
        return SHAPE;
    }
    virtual size_t size() const { return (ROW * COL); }

    __Matrix_impl& operator=(__Matrix_impl& other) {
        assert(this->size() == other.size());
        if (this == &other)
            return *this;
        for (int i = 0 ; i < size() ; i++) {
            std::copy(other.m_data,
                      other.m_data + other.size(),
                      this->m_data);
        }
        return *this;
    }

    friend __Matrix_impl& operator+=(__Matrix_impl& lhs,
                                     const __Matrix_impl& rhs) {
        assert(lhs.get_shape() == rhs.get_shape());
        for (size_t i = 0 ; i < lhs.get_row() ; i++)
            for (size_t j = 0 ; j < lhs.get_col() ; j++)
                lhs(i, j) = lhs(i, j) + rhs(i, j);
        return lhs;
    }

    friend __Matrix_impl& operator+=(__Matrix_impl& lhs,
                                     const T& value) {
        for (int i = 0 ; i < lhs.get_row() ; i++)
            for (size_t j = 0 ; j < lhs.get_col() ; j++)
                lhs(i, j) = lhs(i, j) + value;
        return lhs;
    }

    friend __Matrix_impl& operator+(const __Matrix_impl& lhs,
                                    const __Matrix_impl& rhs) {
        assert(lhs.get_shape() == rhs.get_shape());
        __Matrix_impl *mat = new __Matrix_impl;
        for (int i = 0 ; i < lhs.get_row() ; i++)
            for (size_t j = 0 ; j < lhs.get_col() ; j++)
                (*mat)(i, j) = lhs(i, j) + rhs(i, j);
        return *mat;
    }

    friend __Matrix_impl& operator+(const __Matrix_impl& lhs,
                                    const T& value) {
        __Matrix_impl *mat = new __Matrix_impl;
        for (int i = 0 ; i < lhs.get_row() ; i++)
            for (size_t j = 0 ; j < lhs.get_col() ; j++)
                (*mat)(i, j) = lhs(i, j) + value;
        return *mat;
    }

    friend __Matrix_impl& operator*=(__Matrix_impl& lhs,
                                     const __Matrix_impl& rhs) {
        assert(lhs.get_shape() == rhs.get_shape());
        for (size_t i = 0 ; i < lhs.get_row() ; i++)
            for (size_t j = 0 ; j < lhs.get_col() ; j++)
                lhs(i, j) = lhs(i, j) * rhs(i, j);
        return lhs;
    }

    friend __Matrix_impl& operator*=(__Matrix_impl& lhs,
                                     const T& value) {
        for (int i = 0 ; i < lhs.get_row() ; i++)
            for (size_t j = 0 ; j < lhs.get_col() ; j++)
                lhs(i, j) = lhs(i, j) * value;
        return lhs;
    }

    friend __Matrix_impl& operator*(const __Matrix_impl& lhs,
                                    const __Matrix_impl& rhs) {
        assert(lhs.get_shape() == rhs.get_shape());
        __Matrix_impl *mat = new __Matrix_impl;
        for (size_t i = 0 ; i < lhs.get_row() ; i++)
            for (size_t j = 0 ; j < lhs.get_col() ; j++)
                (*mat)(i, j) = lhs(i, j) * rhs(i, j);
        return *mat;
    }

    friend __Matrix_impl& operator*(const __Matrix_impl& lhs,
                                    const T& value) {
        __Matrix_impl *mat = new __Matrix_impl;
        for (size_t i = 0 ; i < lhs.get_row() ; i++)
            for (size_t j = 0 ; j < lhs.get_col() ; j++)
                (*mat)(i, j) = lhs(i, j) * value;
        return *mat;
    }

    friend __Matrix_impl& operator-=(__Matrix_impl& lhs,
                                     const __Matrix_impl& rhs) {
        assert(lhs.get_shape() == rhs.get_shape());
        for (size_t i = 0 ; i < lhs.get_row() ; i++)
            for (size_t j = 0 ; j < lhs.get_col() ; j++)
                lhs(i, j) = lhs(i, j) - rhs(i, j);
        return lhs;
    }

    friend __Matrix_impl& operator-=(__Matrix_impl& lhs,
                                     const T& value) {
        for (size_t i = 0 ; i < lhs.get_row() ; i++)
            for (size_t j = 0 ; j < lhs.get_col() ; j++)
                lhs(i, j) = lhs(i, j) - value;
        return lhs;
    }

    friend __Matrix_impl& operator-(const __Matrix_impl& lhs,
                                    const __Matrix_impl& rhs) {
        assert(lhs.get_shape() == rhs.get_shape());
        __Matrix_impl *mat = new __Matrix_impl;
        for (size_t i = 0 ; i < lhs.get_row() ; i++)
            for (size_t j = 0 ; j < lhs.get_col() ; j++)
                (*mat)(i, j) = lhs(i, j) - rhs(i, j);
        return *mat;
    }

    friend __Matrix_impl& operator-(const __Matrix_impl& lhs,
                                    const T& value) {
        __Matrix_impl *mat = new __Matrix_impl;
        for (size_t i = 0 ; i < lhs.get_row() ; i++)
            for (size_t j = 0 ; j < lhs.get_col() ; j++)
                (*mat)(i, j) = lhs(i, j) - value;
        return *mat;
    }

    friend __Matrix_impl& operator/=(__Matrix_impl& lhs,
                                     const __Matrix_impl& rhs) {
        assert(lhs.get_shape() == rhs.get_shape());
        for (size_t i = 0 ; i < lhs.get_row() ; i++)
            for (size_t j = 0 ; j < lhs.get_col() ; j++)
                lhs(i, j) = lhs(i, j) / rhs(i, j);
        return lhs;
    }

    friend __Matrix_impl& operator/=(__Matrix_impl& lhs,
                                     const T& value) {
        for (size_t i = 0 ; i < lhs.get_row() ; i++)
            for (size_t j = 0 ; j < lhs.get_col() ; j++)
                lhs(i, j) = lhs(i, j) / value;
        return lhs;
    }

    friend __Matrix_impl& operator/(const __Matrix_impl& lhs,
                                    const __Matrix_impl& rhs) {
        assert(lhs.get_shape() == rhs.get_shape());
        __Matrix_impl *mat = new __Matrix_impl;
        for (size_t i = 0 ; i < lhs.get_row() ; i++)
            for (size_t j = 0 ; j < lhs.get_col() ; j++)
                (*mat)(i, j) = lhs(i, j) / rhs(i, j);
        return *mat;
    }

    friend __Matrix_impl& operator/(const __Matrix_impl& lhs,
                                    const T& value) {
        __Matrix_impl *mat = new __Matrix_impl;
        for (size_t i = 0 ; i < lhs.get_row() ; i++)
            for (size_t j = 0 ; j < lhs.get_col() ; j++)
                (*mat)(i, j) = lhs(i, j) / value;
        return *mat;
    }

    friend __Matrix_impl& operator%=(__Matrix_impl& lhs,
                                     const T& value) {
        for (size_t i = 0 ; i < lhs.get_row() ; i++)
            for (size_t j = 0 ; j < lhs.get_col() ; j++)
                lhs(i, j) = lhs(i, j) % value;
        return lhs;
    }

    friend __Matrix_impl& operator%(const __Matrix_impl& lhs,
                                    const T& value) {
        __Matrix_impl *mat = new __Matrix_impl;
        for (size_t i = 0 ; i < lhs.get_row() ; i++)
            for (size_t j = 0 ; j < lhs.get_col() ; j++)
                (*mat)(i, j) = lhs(i, j) % value;
        return *mat;
    }

    virtual T& operator() (size_t i, size_t j) { return m_data[(i * N) + j]; }
    virtual const T& operator() (size_t i, size_t j) const { return m_data[(i * N) + j];}
    virtual T& diagonal (int i) { return (*this)(i, i); }

    virtual __Vec_impl<T, M> *get_row_vecs() const {
        __Vec_impl<T, M> *vecs = new __Vec_impl<T, M> [get_row()];
        for (size_t i = 0 ; i < get_row() ; i++) {
            size_t begin = (i  * get_col());
            size_t end   = ((i + 1) * get_col());
            T *temp = new T[ get_col()] {0};
            for (size_t j = begin, k = 0 ; j < end && k < get_col() ; j++, k++)
                temp[k] = this->m_data[j];
            vecs[i] = __Vec_impl<T, M>(temp);
        }
        return vecs;
    }

    virtual __Vec_impl<T, N> *get_col_vecs() const {
        __Vec_impl<T, N> *vecs = new __Vec_impl<T, N> [get_col()];
        for (size_t i = 0 ; i < get_col() ; i++) {
            T *temp = new T[get_row()] {0};
            size_t j = i;
            while ( j < this->size() ) {
                temp[static_cast<size_t> (j / get_col())] = this->m_data[j];
                j += get_col();
            }
            vecs[i] = __Vec_impl<T, N>(temp);
        }
        return vecs;
    }

    template<typename F, size_t S,
            std::enable_if_t<std::is_convertible<F, T>::value, bool> = true>
    __Matrix_impl<T, N, S> dot(const __Matrix_impl<F, M, S> *other) const {
        auto this_sh  = this->get_shape();
        auto other_sh = other->get_shape();
        assert(this_sh.col == other_sh.row);

        __Vec_impl<T, M> *this_vecs = this->get_row_vecs();
        __Vec_impl<F, M> *other_vecs = other->get_col_vecs();

        __Matrix_impl<T, N, S> res;
        for (size_t i = 0 ; i < this_sh.row ; i++) {
            for (size_t j = 0 ; j < other_sh.col ; j++) {
                res(i, j) = this_vecs[i].dot(other_vecs[j]);
            }
        }
        return res;
    }

    virtual __Matrix_impl<T, N, M> identity() const {
        static __Matrix_impl<T, N, M> identity_matrix;
        static bool calculate = false;
        if (calculate) return identity_matrix;

        size_t size_identity = min(get_row(), get_col());
        for (size_t i = 0; i < size_identity; i++)
            identity_matrix.diagonal(i) = 1;
        return identity_matrix;
    }

    virtual __Matrix_impl<T, N, M> *transpose() {
        static __Matrix_impl<T, M, N> *transpose_matrix = nullptr;
        if (transpose_matrix != nullptr) return transpose_matrix;

        transpose_matrix = new __Matrix_impl<T, M, N>;
        for (size_t i = 0; i < get_row(); i++) {
            for (size_t j = 0; j < get_col(); j++) {
                (*transpose_matrix)(j,i) = (*this)(i,j);
            }
        }
        return transpose_matrix;
    }

   friend std::ostream& operator<<(std::ostream& os, __Matrix_impl<T, N, M> const & mat) {
       // TODO: Optimize
        for (size_t i = 0; i < mat.get_row(); i++) {
            os << '|';
            for (size_t j = 0; j < mat.get_col(); j++) {
                os << mat(i, j);
                if (j != mat.get_col() - 1) os << ' ';
            }
            os << '|';
            if (i != mat.get_row() - 1) os << '\n';
        }
        return os;
   }

    inline const size_t get_row() const { return ROW; }
    inline const size_t get_col() const { return COL; }
};


template <typename T>
class __Matrix2_impl : public __Matrix_impl<T, 2, 2> {
    using __Matrix_impl<T, 2, 2>::m_data;
    using __Matrix_impl<T, 2, 2>::ROW;
    using __Matrix_impl<T, 2, 2>::COL;

public:
    using __Matrix_impl<T, 2, 2>::__Matrix_impl;
};


template <typename T>
class __Matrix3_impl : public __Matrix_impl<T, 3, 3> {
    using __Matrix_impl<T, 3, 3>::m_data;
    using __Matrix_impl<T, 3, 3>::ROW;
    using __Matrix_impl<T, 3, 3>::COL;
public:
    using __Matrix_impl<T, 3, 3>::__Matrix_impl;
};


template <typename T>
class __Matrix4_impl : public __Matrix_impl<T, 4, 4> {
    using __Matrix_impl<T, 4, 4>::m_data;
    using __Matrix_impl<T, 4, 4>::ROW;
    using __Matrix_impl<T, 4, 4>::COL;
public:
    using __Matrix_impl<T, 4, 4>::__Matrix_impl;
};

};

#endif // GWLA_HPP
