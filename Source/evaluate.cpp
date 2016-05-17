/*
    Copyright (c) 2012, Philipp Krähenbühl
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the Stanford University nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "saliency.h"
#include <cstdio>
#include <vector>
#include <string>
#include "dirent.h"
#include <iostream>

// Uncomment of you want to plot a precision-recall curve
// #define PLOT

#ifdef PLOT
#include "gnuplot_i.hpp"
#endif

#ifdef USE_TBB
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#endif

const std::string ground_truth_path = "data/gt/";
const std::string dataset_path = "data/images/";

// Is the pr curve normalize by the image size (larger images count as much as smaller ones)
const int N_BINS = 256; // Number of bins for the PR curve
const double F_BETA2 = 0.3; // Beta^2 for the f-measure

class Dataset{
	std::vector< std::string > im_, gt_, name_;
public:
	Dataset(){
		DIR *dp = opendir( ground_truth_path.c_str() ); 
		for( dirent *dirp = readdir(dp); dirp != NULL; dirp = readdir(dp) ) {
			std::string name = dirp->d_name;
			if ( name.length() > 3 && name.substr( name.size()-3 ) == "bmp" ) {
				std::string jpgname = name.substr( 0, name.size()-3 ) + "jpg";
				gt_.push_back( ground_truth_path + "/" + name );
				im_.push_back( dataset_path + "/" + jpgname );
				name_.push_back( name.substr( 0, name.size()-4 ) );
			}
		}
		closedir(dp);
	}
	int size() const {
		return gt_.size();
	}
	Mat_<float> gt( int i ) const {
		Mat im = imread( gt_[i], 0 );
		Mat_<float> r;
		// Make the saliency mask binary as it should be!!
		Mat(im > 127).convertTo( r, CV_32F, 1.0 / 255. );
		return r;
	}
	Mat im( int i ) const {
		return imread( im_[i] );
	}
	std::string name( int i ) const {
		return name_[i];
	}
};
static Dataset dataset;

// Collect all the statistics for the precision / recal curves
struct Stat{
	std::vector< long long > bins0, bins1;
	std::vector< double > p_, r_;
	int sm_;
	Stat():bins0( N_BINS, 0 ), bins1( N_BINS, 0 ), p_( N_BINS, 0 ), r_( N_BINS, 0 ), sm_(0) {
	}
	Stat & operator+=( const Stat & o ) {
		for( int i=0; i<N_BINS; i++ ) {
			p_[i] += o.p_[i];
			r_[i] += o.r_[i];
		}
		sm_ += o.sm_;
		return *this;
	}
	void addValue( float v, float gt, float w = 1.0 ){
		int p = v*N_BINS;
		if (p >= N_BINS)
			p = N_BINS;
		
		if (gt < 0.5)
			bins0[ p ] += w;
		else
			bins1[ p ] += w;
	}
	void add( const Mat_<float> & sal, const Mat_<float> & gt ) {
		for( int i=0; i<N_BINS; i++ )
			bins0[i] = bins1[i] = 0;
		
		long long ngt = 0;
		for( int j=0; j<sal.rows; j++ ) {
			const float * psal = sal.ptr<float>( j );
			const float * pgt  = gt .ptr<float>( j );
			for( int i=0; i<sal.cols; i++ ) {
				bool g = pgt[i] > 0.5;
				int s = psal[i]*(N_BINS-1);
				
				bins0[s] += g;
				bins1[s]++;
				ngt += g;
			}
		}
		
		long long nsal = 0, nsal_and_gt = 0;
		for( int i=N_BINS-1; i>=0; i-- ) {
			nsal_and_gt += bins0[i];
			nsal += bins1[i];
			p_[i] += 1.0 * nsal_and_gt / ( nsal + 1e-10);
			r_[i] += 1.0 * nsal_and_gt / ( ngt + 1e-10);
		}
		sm_++;
	}
	std::vector< float > precision() const {
		std::vector< float > r;
		for( int i=0; i<p_.size(); i++ )
			r.push_back( p_[i] / sm_ );
		return r;
	}
	std::vector< float > recall() const {
		std::vector< float > r;
		for( int i=0; i<r_.size(); i++ )
			r.push_back( r_[i] / sm_ );
		return r;
	}
	std::vector< float > f_measure( float beta2 ) const {
		std::vector< float > f( N_BINS ), p = precision(), r = recall();
		for( int i=0; i<N_BINS; i++ )
			f[i] = (1 + beta2) * p[i]*r[i] / (beta2*p[i] + r[i]);
		return f;
	}
};
struct Evaluator{

	const Saliency & saliency_;
	double mae_, p_, r_, f_, cnt_;
	Stat stat_;
	
	Evaluator(const Saliency & saliency):saliency_(saliency),mae_(0),cnt_(0),p_(0),r_(0),f_(0){
	}
	double mae( const Mat_<float> & sal, const Mat_<float> & gt ) const {
		return sum( abs( sal - gt ) )[0] / sal.size().area();
	}
	void evaluate( int i ) {
		Mat im = dataset.im( i );
		Mat_<float> gt = dataset.gt( i );
		
		Mat_<float> sal = saliency_.saliency( im );
// 		Mat_<float> sal;
// 		imread( "data/SF_maps/" + dataset.name(i) + ".jpg", 0 ).convertTo( sal, CV_32F, 1.0 / 255.f );
		
		stat_.add( sal, gt );
		double e = mae( sal, gt );
		
		// Compute the precision and recall with adaptive threshold
		double adaptive_T = 2.0 * sum( sal )[0] / (sal.cols*sal.rows);
		while (sum( sal > adaptive_T )[0] == 0)
			adaptive_T /= 1.2;
		
		double tp = sum( (sal > adaptive_T) & (gt > 0.5) )[0];
		double p = tp / sum( sal > adaptive_T )[0];
		double r = tp / sum( gt > 0.5 )[0];
		double f = (1 + F_BETA2) * p*r / (F_BETA2*p + r + 1e-10);
		
		// Store all the values
		mae_ += e;
		p_ += p;
		r_ += r;
		f_ += f;
		cnt_ += 1.f;
	}
#ifdef USE_TBB
	Evaluator(const Evaluator & o, tbb::split ):saliency_(o.saliency_),mae_(0),cnt_(0),p_(0),r_(0),f_(0){
	}
	void join( const Evaluator & o ) {
		mae_ += o.mae_;
		p_ += o.p_;
		r_ += o.r_;
		f_ += o.f_;
		cnt_ += o.cnt_;
		stat_+= o.stat_;
	}
	void operator()( const tbb::blocked_range<int> & r ) {
		for( int i=r.begin(); i<r.end(); i++ )
			evaluate( i );
	}
	void evalAll() {
		tbb::parallel_reduce( tbb::blocked_range<int>(0, dataset.size() ), *this );
		mae_ /= cnt_;
		p_ /= cnt_;
		r_ /= cnt_;
		f_ /= cnt_;
	}
#else
	void evalAll() {
		for( int i=0; i<dataset.size(); i++ )
			evaluate( i );
		mae_ /= cnt_;
		p_ /= cnt_;
		r_ /= cnt_;
		f_ /= cnt_;
	}
#endif
};

int main( int argc, char * argv[] ) {
	Saliency saliency;
	Evaluator eval( saliency );
	
	eval.evalAll();
	
	printf( "MAE = %f\n", eval.mae_ );
	
	printf("p = %f  r = %f  f = %f\n", eval.p_, eval.r_, eval.f_ );
#ifdef PLOT
	std::vector< float > p = eval.stat_.precision();
	std::vector< float > r = eval.stat_.recall();
	Gnuplot plot("PR");
	plot.set_title( "Precision - Recall" );
	plot.set_style("lines");
	plot.set_grid();
	plot.plot_xy( r, p );
	
	std::cout << std::endl << "Press ENTER to continue..." << std::endl;
    std::cin.clear();
    std::cin.ignore(std::cin.rdbuf()->in_avail());
    std::cin.get();
#endif
	
	return 0;
}
