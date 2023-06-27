#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <map>

/*
    双精度浮点数矩阵的LU分解，QR分解，正交约减（householder, givens），URV分解，解方程组，求行列式
*/

struct Args
{
    char fac_type;
    int m;
    int n;
};

#define M(A_, i_, j_, lda_) (A_)[(i_) + (j_) * (lda_)]
#define EPS 1e-7
#define EQZ(num_) (num_ > -EPS && num_ < EPS)

void printMatrix(double *A, int m, int n, int lda)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (EQZ(M(A, i, j, lda)))
                printf("%3g ", 0.0);
            else
                printf("%3g ", M(A, i, j, lda));
        }
        printf("\n");
    }
    printf("-----------------\n");
}

void copyMatrix(int m, int n, int lda, double *src, double *dest)
{
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++)
        {
            M(dest, i, j, lda) = M(src, i, j, lda);
        }
    }
}

void parseArgs(int argc, char *argv[], Args *args, double **pA, double **pb)
{
    args->fac_type = argv[1][0];
    sscanf(argv[2], "%d", &(args->m));
    sscanf(argv[3], "%d", &(args->n));

    *pA = new double[args->m * args->n]();
    *pb = new double[args->m]();

    std::ifstream fin(argv[4], std::ios::in);

    for (int i = 0; i < args->m; i++)
    {
        for (int j = 0; j < args->n; j++)
        {
            char buf[100];
            if(fin >> buf)
            {
                sscanf(buf, "%lf", &(M(*pA, i, j, args->m)));
            }
        }
    }

    for (int i = 0; i < args->m; i++)
    {
        char buf[100];
        if(fin >> buf)
        {
            sscanf(buf, "%lf", &(M(*pb, i, 0, args->m)));
        }
    }
    
}

void scalStride(int n, double alpha, double *x, int stride_x)
{
    for (int i = 0; i < n; i++)
    {
        *(x + i * stride_x) = alpha * *(x + i *stride_x);
    }
}


void axpyStride(int n, double alpha, double *x, int stride_x, double *y, int stride_y)
{
    for (int i = 0; i < n; i++)
    {
        *(y + i * stride_y) += alpha * *(x + i *stride_x);
    }
}

double dotStride(int n, double *x, int stride_x, double *y, int stride_y)
{
    double res = 0;
    for (int i = 0; i < n; i++)
    {
        res += *(x + i * stride_x) * *(y + i * stride_y);
    }
    return res;
}

void dgemv_n(double *A, int m, int n, int lda, double alpha, double beta, double *x, int stride_x, double *y, int stride_y)
{

    // scalStride(m, beta, y, stride_y);

    // for (int j = 0; j < n; j++)
    // {
    //     axpyStride(m, alpha * *(x + j * stride_x), &(M(A, 0, j, lda)), 1, y, stride_y);
    // }
    for (int i = 0; i < m; i++)
    {
        *(y + i * stride_y) = beta * *(y + i * stride_y) + alpha * dotStride(n, &(M(A, i, 0, lda)), lda, x, stride_x);
    }

}

void dgemv_t(double *A, int m, int n, int lda, double alpha, double beta, double *x, int stride_x, double *y, int stride_y)
{
    for (int i = 0; i < m; i++)
    {
        y[i] = beta * y[i] + alpha * dotStride(n, &(M(A, 0, i, lda)), 1, x, stride_x);
    }
}

double normStride(int n, double *x, int stride_x)
{
    return sqrt(dotStride(n, x, stride_x, x, stride_x));
}

void forwardSolveL(double *L, int n, int lda, double *x, double *b)
{
    for (int i = 0; i < n; i++)
    {
        x[i] = b[i] - dotStride(i, &(M(L, i, 0, lda)), lda, x, 1);
    }
}

void backwardSolveU(double *U, int n, int lda, double *x, double *b)
{
    for (int i = n-1; i >= 0; i--)
    {
        x[i] = (b[i] - dotStride(n - i - 1, &(M(U, i, i+1, lda)), lda, x + i + 1, 1)) / M(U, i, i, lda);
    }
}

void LU_fact(double *A, int n, int lda)
{
    if (n <= 1) return; 
    if (EQZ(M(A, 0, 0, lda))) 
    {
        printf("pivot can't be 0.0, LU failed!\n");
        return;
    }
    for (int i = 1; i < n; i++)
    {
        M(A, i, 0, lda) = M(A, i, 0, lda) / M(A, 0, 0, lda);

        axpyStride(n-1, -M(A, i, 0, lda), &(M(A, 0, 1, lda)), lda, &(M(A, i, 1, lda)), lda);

    }

    LU_fact(A + 1 + lda, n-1, lda);
}

void QR_fact(double *A, int m, int n, int lda, double *R)
{
    for (int j = 0; j < n; j++)
    {
        for (int k = 0; k < j; k++)
        {
            M(R, k, j, n) = dotStride(m, &(M(A, 0, j, lda)), 1, &(M(A, 0, k, lda)), 1);
        }

        dgemv_n(A, m, j, lda, -1, 1, &(M(R, 0, j, n)), 1, &(M(A, 0, j, lda)), 1);

        M(R, j, j, n) = normStride(m, &(M(A, 0, j, lda)), 1);

        scalStride(m, 1 / M(R, j, j, n), &(M(A, 0, j, lda)), 1);
    }
}

void genReflector(int n, double *x, double *R)
{
    double *u = new double[n]();
    copyMatrix(n, 1, n, x, u);
    u[0] -= normStride(n, x, 1);

    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < n; i++)
        {
            M(R, i, j, n) = -2 * (u[i] * u[j]) / dotStride(n, u, 1, u, 1);
            if (i == j)
                M(R, i, j, n) += 1;
        }
    }

    delete []u;
}

void hhReduction(double *A, int m, int n, int lda, double *R)
{
    double *bak = new double[m]();
    for (int j = 0; j < n; j++)
    {
        copyMatrix(m, 1, lda, &(M(A, 0, j, lda)), bak);
        dgemv_n(R, m, m, m, 1, 0, bak, 1, &(M(A, 0, j, lda)), 1);
    }
    delete []bak;
}


void mergeReflector(double *R, int m, double *P, int pm)
{
    if (pm == m)
    {
        copyMatrix(m, m, m, R, P);
        return;
    }
    double *P3 = &(M(P, pm-m, 0, pm));
    double *P4 = &(M(P, pm-m, pm-m, pm));

    hhReduction(P3, m, pm-m, pm, R);
    hhReduction(P4, m, m, pm, R);
}

void houseHolderReduction(double *A, int m, int n, int lda, double *P, int pm)
{
    if (m <= 1) return;
    if (n <= 0 ) return;
    double *reflector = new double[m * m]();

    genReflector(m, &(M(A, 0, 0, lda)), reflector);

    hhReduction(A, m, n, lda, reflector);

    mergeReflector(reflector, m, P, pm);

    delete []reflector;

    houseHolderReduction(&(M(A, 1, 1, lda)), m-1, n-1, lda, P, pm);    
}

void givensRotation(double c, double s, int i, int j, double *x)
{
    double t1 = c * x[i] + s * x[j];
    double t2 = -s * x[i] + c * x[j];

    x[i] = t1;
    x[j] = t2;

}

void givensReduction(double *A, int m, int n, int lda, double *P)
{
    for (int j = 0; j < n; j++)
    {
        for (int i = j+1; i < m; i++)
        {
            double t = sqrt(M(A, j, j, lda) * M(A, j, j, lda) + M(A, i, j, lda) * M(A, i, j, lda));
            double c = M(A, j, j, lda) / t;
            double s = M(A, i, j, lda) / t;

            for (int jj = 0; jj < n; jj++)
            {
                givensRotation(c, s, j, i, &(M(A, 0, jj, lda)));
                givensRotation(c, s, j, i, &(M(P, 0, jj, lda)));
            }
            
        }
    }
}

void rowSwap(double *A, int m ,int n, int lda, int i, int j)
{
    if (j >= 0 && i >= 0 && i < m && j < m && i != j)
    {
        for (int k = 0; k < n; k++)
        {
            double t = M(A, i, k, lda);
            M(A, i, k, lda) = M(A, j, k, lda);
            M(A, j, k, lda) = t; 
        }
    }
}

void rowScal(double *A, int m, int n, int lda, int i, double alpha)
{
    if (i < m && i >= 0)
        scalStride(n, alpha, &(M(A, i, 0, lda)), lda);
}

void rowScalAdd(double *A, int m, int n, int lda, int i, int j, double alpha)
{
    if (j >= 0 && i >= 0 && i < m && j < m && i != j)
    {
        axpyStride(n, alpha, &(M(A, i, 0, lda)), lda, &(M(A, j, 0, lda)), lda);
    }
}


int gaussJord(double *A, int m, int n, int lda)
{
    int p_row = 0;
    for (int j = 0; j < n && p_row < m; j++)
    {
        
        for (int ii = p_row; ii < m; ii++)
        {
            if (!EQZ(M(A, ii, j, lda)))
            {
                rowSwap(A, m, n, lda, ii, p_row);
            }
        }
        double pivot = M(A, p_row, j, lda);
        if (EQZ(pivot)) continue;

        rowScal(A, m, n, lda, p_row, 1 / pivot);

        for (int i = 0; i < m; i++)
        {
            if (i != p_row)
            {
                rowScalAdd(A, m, n, lda, p_row, i, -M(A, i, j, lda));
            }
        }
        p_row++;
    }
    return p_row;    
}

void genUV(double *A, int m, int n, int lda, int rank, double *U, double *V)
{
    int U_col = 0;
    int V_col = rank;
    
    std::vector<int> basicCol;
    std::vector<int> nonBasicCol;
    std::map<int, int> nonMap;

    for (int i = 0; i < m; i++)
    {
        int j = 0;
        for (; j < n; j++)
        {
            if (!EQZ(M(A, i, j, lda)))
            {
                copyMatrix(m, 1, lda, &(M(A, 0, j, lda)), &(M(U, 0, U_col++, m)));
                basicCol.push_back(j);
                break;
            }
        }
    }

    for (int j = 0; j < n; j++)
    {
        bool flag = false;
        for (auto b_c : basicCol)
        {
            if (b_c == j)
            {
                flag = true;
            }
        }

        if (!flag)
        {
            nonBasicCol.push_back(j);
            nonMap[j] = V_col++;
            M(V, j, nonMap[j], n) = 1;
        }
    }

    for (int i = 0; i < rank; i++)
    {
        for (int j = basicCol[i] + 1; j < n; j++)
        {
            if (!EQZ(M(A, i, j, lda)))
            {
                M(V, basicCol[i], nonMap[j], n) = -M(A, i, j, lda);
            }
        }
    }
}

void trans(double *A, int m, int n, int lda, double *AT, int ldaT)
{
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++)
        {
            M(AT, j, i, ldaT) = M(A, i, j, lda);
        }
    }
}

void ortUV(int m, int n, int rank, double *U, double *V)
{
    double *R = new double[std::max(m, n) * std::max(m, n)]();
    
    QR_fact(&(M(U, 0, rank, m)), m, m - rank, m, R);
    QR_fact(&(M(V, 0, rank, n)), n, n - rank, n, R);
}


void URV_fact(double *A, int m, int n, int lda, double *U, double *R, double *V)
{
    double *A_bak = new double[m * n]();
    copyMatrix(m, n, lda, A, A_bak);
    copyMatrix(m, n, lda, A, R);

    int rank = gaussJord(A, m, n, lda);

    printf("Row form A (rank : %d) = \n", rank);
    printMatrix(A, m, n, lda);

    genUV(A, m, n, lda, rank, U, V);

    double *AT = new double[n * m]();
    trans(A_bak, m, n, lda, AT, n);
    printf("AT = \n");
    printMatrix(AT, n, m, n);

    gaussJord(AT, n, m, n);

    printf("Row form AT (rank : %d) = \n", rank);
    printMatrix(AT, n, m, n);

    genUV(AT, n, m, n, rank, V, U);

    ortUV(m, n, rank, U, V);

    for (int j = 0; j < n; j++)
    {
        dgemv_t(U, m, m, m, 1, 0, &(M(R, 0, j, lda)), 1, &(M(AT, 0, j, lda)), 1);
    }

    for (int j = 0; j < n; j++)
    {
        dgemv_n(AT, m, n, lda, 1, 0, &(M(V, 0, j, n)), 1, &(M(R, 0, j, lda)), 1);
    }

    delete []A_bak;
    delete []AT;

}

int getSign(int index[], int n)
{
    int sign = 1;
    for (int i = 0; i < n-1; i++)
    {
        for (int j = i+1; j < n; j++)
        {
            if(index[i] > index[j])
            {
                sign = -sign;
            }
        }
        
    }
    return sign;
}

void getDet(double *A, int n, int lda, int depth, double tempRes, bool flag[], int index[], double &res)
{
    if(depth == n)
    {
        res += getSign(index, n) * tempRes;
        return;
    }

    for (int i = 0; i < n; i++)
    {
        if(!flag[i])
        {
            flag[i] = true;
            index[depth] = i;
            getDet(A, n, lda, depth+1, tempRes*M(A, depth, i, lda), flag, index, res);
            flag[i] = false;
        }
    }
}

double det(double *A, int n, int lda)
{
    double res = 0;
    bool flag[1000] = {false};
    int index[1000];
    getDet(A, n, lda, 0, 1, flag, index, res);
    return res;
}

void LUFactAndSolve(double *A, int m, int n, int lda, double *b)
{
    if (m != n) 
    {
        printf("not square.\n");
        return;
    }
    LU_fact(A, n, lda);

    printf("LU = \n");
    printMatrix(A, m, n, lda);

    double *x = new double[n]();

    forwardSolveL(A, n, lda, x, b);

    printf("y = \n");
    printMatrix(x, n, 1, lda);

    backwardSolveU(A, n, lda, b, x);
    
    printf("x = \n");
    printMatrix(b, n, 1, lda);


    delete []x;
}

void QRFactAndSolve(double *A, int m, int n, int lda, double *b)
{
    
    double *R = new double[n * n]();

    QR_fact(A, m, n, lda, R);

    printf("Q = \n");
    printMatrix(A, m, n, lda);

    printf("R = \n");
    printMatrix(R, n, n, n);

    if (m != n)
    {
        printf("not square, can't solve\n");
        delete []R;
        return;
    }

    double *x = new double[n]();
    dgemv_t(A, n, n, lda, 1, 0, b, 1, x, 1);

    printf("Q^Tb = \n");
    printMatrix(x, n, 1, n);

    backwardSolveU(R, n, n, b, x);

    printf("x = \n");
    printMatrix(b, n, 1, n);


    delete []x;
    delete []R;
}

void houseHolderReAndSolve(double *A, int m, int n, int lda, double *b)
{
    double *P = new double[m * m]();

    houseHolderReduction(A, m, n, lda, P, m);

    printf("T = \n");
    printMatrix(A, m, n, lda);

    printf("P = \n");
    printMatrix(P, m, m, m);

    double *b_bak = new double[m]();
    copyMatrix(m, 1, m, b, b_bak);

    dgemv_n(P, m, m, m, 1, 0, b_bak, 1, b, 1);
    delete []b_bak;
    delete []P;

    printf("Pb = \n");
    printMatrix(b, m, 1, m);

    if (m != n)
    {
        printf("not square, can't solve\n");
        return;
    }

    backwardSolveU(A, n, m, b, b);
    printf("x = \n");
    printMatrix(b, m, 1, lda);   
}

void setI(double *P, int n, int lda)
{
    for (int i = 0; i < n; i++)
    {
        M(P, i, i, lda) = 1;
        for (int j = 0; j < n; j++)
        {
            if (i != j) M(P, i, j, lda) = 0;
        }
        
    }
}

void givensReAndSolve(double *A, int m, int n, int lda, double *b)
{
    if (m != n)
    {
        printf("not square.\n");
        return;
    }

    double *P = new double[m * n]();

    setI(P, n, n);

    givensReduction(A, m, n, lda, P);

    printf("T = \n");
    printMatrix(A, m, n, lda);
    
    printf("P = \n");
    printMatrix(P, m, n, lda);

    double *b_bak = new double[m]();
    copyMatrix(m, 1, m, b, b_bak);

    dgemv_n(P, m, m, m, 1, 0, b_bak, 1, b, 1);
    delete []b_bak;
    delete []P;

    printf("Pb = \n");
    printMatrix(b, m, 1, m);

    backwardSolveU(A, n, m, b, b);
    printf("x = \n");
    printMatrix(b, m, 1, lda);
}

void URVFactAndSolve(double *A, int m, int n, int lda, double *b)
{
    double *U = new double[m * m]();
    double *R = new double[m * n]();
    double *V = new double[n * n]();
    
    URV_fact(A, m, n, lda, U, R, V);

    printf("U = \n");
    printMatrix(U, m, m, m);

    printf("V = \n");
    printMatrix(V, n, n, n);

    printf("R = \n");
    printMatrix(R, m, n, lda);

    printf("I don't know how to use URV factorization to solve Ax=b.\n");

    delete []U;
    delete []R;
    delete []V;
}

void verify(double *A, int m, int n, int lda, double *x, double *b)
{
    dgemv_n(A, m, n, lda, 1, -1, x, 1, b, 1);
    for (int i = 0; i < m; i++)
    {
        if (!EQZ(b[i]))
        {
            printf("%lf FAIL.\n", b[i]);
            return;
        }
    }
    printf("PASS.\n");
}

int main(int argc, char *argv[])
{
    Args args;

    double *A, *b; 

    parseArgs(argc, argv, &args, &A, &b);

    double *A_ref = new double[args.m * args.n]();
    double *b_ref = new double[args.m]();


    copyMatrix(args.m, args.n, args.m, A, A_ref);
    copyMatrix(args.m, 1, args.m, b, b_ref);

    printf("A = \n");
    printMatrix(A, args.m, args.n, args.m);

    printf("b = \n");
    printMatrix(b, args.m, 1, args.m);

    printf("det(A) = %lf\n", det(A, std::min(args.m, args.n), args.m));
    
    switch (args.fac_type)
    {
    case '0':
        LUFactAndSolve(A, args.m, args.n, args.m, b);
        break;
    case '1':
        QRFactAndSolve(A, args.m, args.n, args.m, b);
        break;
    case '2':
        houseHolderReAndSolve(A, args.m, args.n, args.m, b);
        break;
    case '3':
        givensReAndSolve(A, args.m, args.n, args.m, b);
        break;
    case '4':
        URVFactAndSolve(A, args.m, args.n, args.m, b);
        return 0;
        break;
    default:
        break;
    }

    verify(A_ref, args.m, args.n, args.m, b, b_ref);

    delete []A;
    delete []b;
    delete []A_ref;
    delete []b_ref;

    return 0;
}