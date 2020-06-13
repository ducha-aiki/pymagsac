#pragma once
#include <opencv2/core.hpp>
#include <iostream>
using namespace cv;
/*
  Finds real roots of cubic, quadratic or linear equation.
  The original code has been taken from Ken Turkowski web page
  (http://www.worldserver.com/turk/opensource/) and adopted for OpenCV.
  Here is the copyright notice.

  -----------------------------------------------------------------------
  Copyright (C) 1978-1999 Ken Turkowski. <turk@computer.org>

    All rights reserved.

    Warranty Information
      Even though I have reviewed this software, I make no warranty
      or representation, either express or implied, with respect to this
      software, its quality, accuracy, merchantability, or fitness for a
      particular purpose.  As a result, this software is provided "as is,"
      and you, its user, are assuming the entire risk as to its quality
      and accuracy.

    This code may be used and freely distributed as long as it includes
    this copyright notice and the above warranty information.
  -----------------------------------------------------------------------
*/

int solveCubic( Mat& coeffs, Mat& roots )
{

    const int n0 = 3;
    int ctype = coeffs.type();

    CV_Assert( ctype == CV_32F || ctype == CV_64F );
    CV_Assert( (coeffs.size() == Size(n0, 1) ||
                coeffs.size() == Size(n0+1, 1) ||
                coeffs.size() == Size(1, n0) ||
                coeffs.size() == Size(1, n0+1)) );

    
    int i = -1, n = 0;
    double a0 = 1., a1, a2, a3;
    double x0 = 0., x1 = 0., x2 = 0.;
    int ncoeffs = coeffs.rows + coeffs.cols - 1;

    if( ctype == CV_32FC1 )
    {
        if( ncoeffs == 4 )
            a0 = coeffs.at<float>(++i);

        a1 = coeffs.at<float>(i+1);
        a2 = coeffs.at<float>(i+2);
        a3 = coeffs.at<float>(i+3);
    }
    else
    {
        if( ncoeffs == 4 )
            a0 = coeffs.at<double>(++i);

        a1 = coeffs.at<double>(i+1);
        a2 = coeffs.at<double>(i+2);
        a3 = coeffs.at<double>(i+3);
    }


    if( a0 == 0 )
    {
        if( a1 == 0 )
        {
            if( a2 == 0 )
                n = a3 == 0 ? -1 : 0;
            else
            {
                // linear equation
                x0 = -a3/a2;
                n = 1;
            }
        }
        else
        {
            // quadratic equation
            double d = a2*a2 - 4*a1*a3;
            if( d >= 0 )
            {
                d = std::sqrt(d);
                double q1 = (-a2 + d) * 0.5;
                double q2 = (a2 + d) * -0.5;
                if( fabs(q1) > fabs(q2) )
                {
                    x0 = q1 / a1;
                    x1 = a3 / q1;
                }
                else
                {
                    x0 = q2 / a1;
                    x1 = a3 / q2;
                }
                n = d > 0 ? 2 : 1;
            }
        }
    }
    else
    {
        a0 = 1./a0;
        a1 *= a0;
        a2 *= a0;
        a3 *= a0;

        double Q = (a1 * a1 - 3 * a2) * (1./9);
        double R = (2 * a1 * a1 * a1 - 9 * a1 * a2 + 27 * a3) * (1./54);
        double Qcubed = Q * Q * Q;
        double d = Qcubed - R * R;

        if( d > 0 )
        {
            double theta = acos(R / sqrt(Qcubed));
            double sqrtQ = sqrt(Q);
            double t0 = -2 * sqrtQ;
            double t1 = theta * (1./3);
            double t2 = a1 * (1./3);
            x0 = t0 * cos(t1) - t2;
            x1 = t0 * cos(t1 + (2.*CV_PI/3)) - t2;
            x2 = t0 * cos(t1 + (4.*CV_PI/3)) - t2;
            n = 3;
        }
        else if( d == 0 )
        {
            if(R >= 0)
            {
                x0 = -2*pow(R, 1./3) - a1/3;
                x1 = pow(R, 1./3) - a1/3;
            }
            else
            {
                x0 = 2*pow(-R, 1./3) - a1/3;
                x1 = -pow(-R, 1./3) - a1/3;
            }
            x2 = 0;
            n = x0 == x1 ? 1 : 2;
            x1 = x0 == x1 ? 0 : x1;
        }
        else
        {
            double e;
            d = sqrt(-d);
            e = pow(d + fabs(R), 1./3);
            if( R > 0 )
                e = -e;
            x0 = (e + Q / e) - a1 * (1./3);
            n = 1;
        }
    }

    if( roots.type() == CV_32FC1 )
    {
        roots.at<float>(0) = (float)x0;
        roots.at<float>(1) = (float)x1;
        roots.at<float>(2) = (float)x2;
    }
    else
    {
        roots.at<double>(0) = x0;
        roots.at<double>(1) = x1;
        roots.at<double>(2) = x2;
    }

    return n;
}


