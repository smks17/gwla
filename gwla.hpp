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

template <typename T, size_t N>
class __Vec_impl {
protected:
    static const size_t m_size = N;
    T * m_data;

public:
    __Vec_impl() { m_data = reinterpret_cast<T*>( new T[m_size]() ); };
    template <typename S>
    __Vec_impl(S value) {
        m_data = reinterpret_cast<T*>( new T[m_size] );
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

    T dot(const __Vec_impl other) {
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

    inline const size_t size() const { return m_size; }
};


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


typedef Vec<2>                Vec2;
typedef Vec_i8<2>             Vec2_i8;
typedef Vec_i16<2>            Vec2_i16;
typedef Vec_i32<2>            Vec2_i32;
typedef Vec_u8<2>             Vec2_u8;
typedef Vec_u16<2>            Vec2_u16;
typedef Vec_u32<2>            Vec2_u32;
typedef Vec_d<2>              Vec2_d;


typedef Vec<3>                Vec3;
typedef Vec_i8<3>             Vec3_i8;
typedef Vec_i16<3>            Vec3_i16;
typedef Vec_i32<3>            Vec3_i32;
typedef Vec_u8<3>             Vec3_u8;
typedef Vec_u16<3>            Vec3_u16;
typedef Vec_u32<3>            Vec3_u32;
typedef Vec_d<3>              Vec3_d;


typedef Vec<4>                Vec4;
typedef Vec_i8<4>             Vec4_i8;
typedef Vec_i16<4>            Vec4_i16;
typedef Vec_i32<4>            Vec4_i32;
typedef Vec_u8<4>             Vec4_u8;
typedef Vec_u16<4>            Vec4_u16;
typedef Vec_u32<4>            Vec4_u32;
typedef Vec_d<4>              Vec4_d;


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
    template<typename S>
    explicit __Matrix_impl(S value) {
        m_data = reinterpret_cast<T*>( new T[this->size()] {value} );
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


typedef Mat<2,2>                Mat2;
typedef Mat_i8<2,2>             Mat2_i8;
typedef Mat_i16<2,2>            Mat2_i16;
typedef Mat_i32<2,2>            Mat2_i32;
typedef Mat_u8<2,2>             Mat2_u8;
typedef Mat_u16<2,2>            Mat2_u16;
typedef Mat_u32<2,2>            Mat2_u32;
typedef Mat_d<2,2>              Mat2_d;

typedef Mat<3,3>                Mat3;
typedef Mat_i8<3,3>             Mat3_i8;
typedef Mat_i16<3,3>            Mat3_i16;
typedef Mat_i32<3,3>            Mat3_i32;
typedef Mat_u8<3,3>             Mat3_u8;
typedef Mat_u16<3,3>            Mat3_u16;
typedef Mat_u32<3,3>            Mat3_u32;
typedef Mat_d<3,3>              Mat3_d;

typedef Mat<4,4>                Mat4;
typedef Mat_i8<4,4>             Mat4_i8;
typedef Mat_i16<4,4>            Mat4_i16;
typedef Mat_i32<4,4>            Mat4_i32;
typedef Mat_u8<4,4>             Mat4_u8;
typedef Mat_u16<4,4>            Mat4_u16;
typedef Mat_u32<4,4>            Mat4_u32;
typedef Mat_d<4,4>              Mat4_d;

};

#endif // GWLA_HPP
