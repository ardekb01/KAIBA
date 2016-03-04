// BUG FIX (2011-07-29): The default interpolation method was nearest neighbor. The -nn option would use
// trilinear interpolatin.  This was corrected.  Now the default interpolation method is trilinear and 
// the -nn option forces the program to use nearest neighbor.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <volume.h>
#include <ctype.h>

#include <nifti1_io.h>
#include <niftiimage.h>
#include "babak_lib.h"
#include <sph.h>
#include <smooth.h>
#include <landmarks.h>

#define YES 1
#define NO 0
#define HISTCUTOFF 0.25
#define MAXNCLASS 15
#define MXFRAC 0.4
#define MXFRAC2 0.2

// Won't be re-defined if this variable is defined at compilation time
#ifndef MAXITER
#define MAXITER 20
#endif

// PIL brain cloud threshold for definining a brain mask
#ifndef CLOUD_THRESH
#define CLOUD_THRESH 90 
#endif

#ifndef TOLERANCE
#define TOLERANCE 1.e-7
#endif

int opt;

/////////////////////////////////////////////////////////////////////////
// Global variables

int opt_png=NO; // flag for outputing PNG images
int opt_v=NO; // flag for verbose mode
int opt_newPIL=YES;

/////////////////////////////////////////////////////////////////////////

static struct option options[] =
{
   {"-v",0,'v'},
   {"-version",0,'V'},
   {"-V",0,'V'},
   {"-png",0,'g'},
   {"-oldPIL",0,'n'},
   {"-p",1,'p'},   // output prefix
   {"-b",1,'b'},   // baseline image
   {"-f",1,'f'},   // follow-up image
   {"-blm",1,'l'},  // baseline landmark 
   {"-flm",1,'m'},  // folow-up landmark 
   {0,0,0}
};

void print_help_and_exit()
{
   printf("\nUsage: kaiba [options] -p <prefix> -b <basline>.nii [-f <follow-up>.nii]\n"
   "\nRequired arguments:\n"
   "   -p <prefix>: Output files prefix\n"
   "   -b <basline>.nii: Baseline T1W volume (NIFTI format)\n"
   "\nOptions:\n"
   "   -v : Enables verbose mode\n"
   "   -V or -version : Prints program version\n"
   "   -png : Outputs images in PNG format in addition to PPM\n"
   "   -f <follow-up>.nii: Follow-up T1W volume (NIFTI format)\n"
   "   -blm <filename>: Manually specifies AC/PC/RP landmarks at baseline\n"
   "   -flm <filename>: Manually specifies AC/PC/RP landmarks at follow-up\n"
   "\n");

   exit(0);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
double ssd_cost_function(float *T, DIM dimb, DIM dimf, float *sclbim, float *sclfim, short *bmsk, short *fmsk)
{
   int kmin_f=0; 
   int kmax_f=dimf.nz-1;
   int jmin_f=0; 
   int jmax_f=dimf.ny-1;
   int imin_f=0; 
   int imax_f=dimf.nx-1;
   int kmin_b=0;
   int kmax_b=dimb.nz-1;
   int jmin_b=0;
   int jmax_b=dimb.ny-1;
   int imin_b=0;
   int imax_b=dimb.nx-1;

   float *invT;
   float dif;
   double cost=0.0;
   int v, slice_offset, offset;
   float Tmod[16]; //modified T
   float invTmod[16]; //modified invT

   float psub0, psub1, psub2;
   float nxsub2, nysub2, nzsub2; 

   float ptrg0, ptrg1, ptrg2;
   float nxtrg2, nytrg2, nztrg2;

   invT = inv4(T);

   nxsub2 = (dimf.nx-1)/2.0;
   nysub2 = (dimf.ny-1)/2.0;
   nzsub2 = (dimf.nz-1)/2.0;

   nxtrg2 = (dimb.nx-1)/2.0;
   nytrg2 = (dimb.ny-1)/2.0;
   nztrg2 = (dimb.nz-1)/2.0;

   ////////////////////////////////////////
   Tmod[0] = T[0]*dimf.dx/dimb.dx;
   Tmod[1] = T[1]*dimf.dy/dimb.dx;
   Tmod[2] = T[2]*dimf.dz/dimb.dx;
   Tmod[3] = T[3]/dimb.dx + nxtrg2;

   Tmod[4] = T[4]*dimf.dx/dimb.dy;
   Tmod[5] = T[5]*dimf.dy/dimb.dy;
   Tmod[6] = T[6]*dimf.dz/dimb.dy;
   Tmod[7] = T[7]/dimb.dy + nytrg2;

   Tmod[8] = T[8]*dimf.dx/dimb.dz;
   Tmod[9] = T[9]*dimf.dy/dimb.dz;
   Tmod[10] = T[10]*dimf.dz/dimb.dz;
   Tmod[11] = T[11]/dimb.dz + nztrg2;
   ////////////////////////////////////////
   invTmod[0] = invT[0]*dimb.dx/dimf.dx;
   invTmod[1] = invT[1]*dimb.dy/dimf.dx;
   invTmod[2] = invT[2]*dimb.dz/dimf.dx;
   invTmod[3] = invT[3]/dimf.dx + nxsub2;

   invTmod[4] = invT[4]*dimb.dx/dimf.dy;
   invTmod[5] = invT[5]*dimb.dy/dimf.dy;
   invTmod[6] = invT[6]*dimb.dz/dimf.dy;
   invTmod[7] = invT[7]/dimf.dy + nysub2;

   invTmod[8] = invT[8]*dimb.dx/dimf.dz;
   invTmod[9] = invT[9]*dimb.dy/dimf.dz;
   invTmod[10] = invT[10]*dimb.dz/dimf.dz;
   invTmod[11] = invT[11]/dimf.dz + nzsub2;
   ////////////////////////////////////////
   
   float t2, t6, t10;
   float t1, t5, t9;

   for(int k=kmin_f; k<=kmax_f; k++)
   {
      slice_offset = k*dimf.np;
      psub2 = (k-nzsub2);
      t2  = Tmod[2]*psub2  + Tmod[3];
      t6  = Tmod[6]*psub2  + Tmod[7];
      t10 = Tmod[10]*psub2 + Tmod[11];
      for(int j=jmin_f; j<=jmax_f; j++)
      {
         offset = slice_offset + j*dimf.nx;
         psub1 = (j-nysub2);
         t1 = Tmod[1]*psub1;
         t5 = Tmod[5]*psub1;
         t9 = Tmod[9]*psub1;
         for(int i=imin_f; i<=imax_f; i++)
         {
            v = offset + i;

            if( fmsk[v]>0)
            {
               psub0 = (i-nxsub2);

               ptrg0 = Tmod[0]*psub0 + t1 + t2;
               ptrg1 = Tmod[4]*psub0 + t5 + t6;
               ptrg2 = Tmod[8]*psub0 + t9 + t10;

               dif = sclfim[v] - linearInterpolator(ptrg0, ptrg1, ptrg2, sclbim, dimb.nx, dimb.ny, dimb.nz, dimb.np);
               cost += (dif*dif);
            }
         }
      }
   }

   for(int k=kmin_b; k<=kmax_b; k++)
   {
      slice_offset = k*dimb.np;
      ptrg2 = (k-nztrg2);
      t2  = invTmod[2]*ptrg2  + invTmod[3];
      t6  = invTmod[6]*ptrg2  + invTmod[7];
      t10 = invTmod[10]*ptrg2 + invTmod[11];
      for(int j=jmin_b; j<=jmax_b; j++)
      {
         offset = slice_offset + j*dimb.nx;
         ptrg1 = (j-nytrg2);
         t1 = invTmod[1]*ptrg1;
         t5 = invTmod[5]*ptrg1;
         t9 = invTmod[9]*ptrg1;
         for(int i=imin_b; i<=imax_b; i++)
         {
            v = offset + i;

            if( bmsk[v]>0)
            {
               ptrg0 = (i-nxtrg2);

               psub0 = invTmod[0]*ptrg0 + t1 + t2;
               psub1 = invTmod[4]*ptrg0 + t5 + t6;
               psub2 = invTmod[8]*ptrg0 + t9 + t10;

               dif = sclbim[v] - linearInterpolator(psub0, psub1, psub2, sclfim, dimf.nx, dimf.ny, dimf.nz, dimf.np);
               cost += (dif*dif);
            }
         }
      }
   }

   free(invT);
   return(cost);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

double ncc_cost_function(float *T, DIM dimb, DIM dimf, float *sclbim, float *sclfim, short *bmsk, short *fmsk)
{
   int kmin_f=0; 
   int kmax_f=dimf.nz-1;
   int jmin_f=0; 
   int jmax_f=dimf.ny-1;
   int imin_f=0; 
   int imax_f=dimf.nx-1;
   int kmin_b=0;
   int kmax_b=dimb.nz-1;
   int jmin_b=0;
   int jmax_b=dimb.ny-1;
   int imin_b=0;
   int imax_b=dimb.nx-1;

   int n=0; // number of voxels that take part in the NCC calculation
   double sum1, sum2, sum11, sum22, sum12;
   float subject_image_value, target_image_value;

   float *invT;
   float dif;
   double cost=0.0;
   int v, slice_offset, offset;
   float Tmod[16]; //modified T
   float invTmod[16]; //modified invT

   float psub0, psub1, psub2;
   float nxsub2, nysub2, nzsub2; 

   float ptrg0, ptrg1, ptrg2;
   float nxtrg2, nytrg2, nztrg2;

   invT = inv4(T);

   nxsub2 = (dimf.nx-1)/2.0;
   nysub2 = (dimf.ny-1)/2.0;
   nzsub2 = (dimf.nz-1)/2.0;

   nxtrg2 = (dimb.nx-1)/2.0;
   nytrg2 = (dimb.ny-1)/2.0;
   nztrg2 = (dimb.nz-1)/2.0;

   ////////////////////////////////////////
   Tmod[0] = T[0]*dimf.dx/dimb.dx;
   Tmod[1] = T[1]*dimf.dy/dimb.dx;
   Tmod[2] = T[2]*dimf.dz/dimb.dx;
   Tmod[3] = T[3]/dimb.dx + nxtrg2;

   Tmod[4] = T[4]*dimf.dx/dimb.dy;
   Tmod[5] = T[5]*dimf.dy/dimb.dy;
   Tmod[6] = T[6]*dimf.dz/dimb.dy;
   Tmod[7] = T[7]/dimb.dy + nytrg2;

   Tmod[8] = T[8]*dimf.dx/dimb.dz;
   Tmod[9] = T[9]*dimf.dy/dimb.dz;
   Tmod[10] = T[10]*dimf.dz/dimb.dz;
   Tmod[11] = T[11]/dimb.dz + nztrg2;
   ////////////////////////////////////////
   invTmod[0] = invT[0]*dimb.dx/dimf.dx;
   invTmod[1] = invT[1]*dimb.dy/dimf.dx;
   invTmod[2] = invT[2]*dimb.dz/dimf.dx;
   invTmod[3] = invT[3]/dimf.dx + nxsub2;

   invTmod[4] = invT[4]*dimb.dx/dimf.dy;
   invTmod[5] = invT[5]*dimb.dy/dimf.dy;
   invTmod[6] = invT[6]*dimb.dz/dimf.dy;
   invTmod[7] = invT[7]/dimf.dy + nysub2;

   invTmod[8] = invT[8]*dimb.dx/dimf.dz;
   invTmod[9] = invT[9]*dimb.dy/dimf.dz;
   invTmod[10] = invT[10]*dimb.dz/dimf.dz;
   invTmod[11] = invT[11]/dimf.dz + nzsub2;
   ////////////////////////////////////////
   
   float t2, t6, t10;
   float t1, t5, t9;

   // initialize sums to zero
   sum1=sum2=sum11=sum22=sum12=0.0;

   for(int k=kmin_f; k<=kmax_f; k++)
   {
      slice_offset = k*dimf.np;
      psub2 = (k-nzsub2);
      t2  = Tmod[2]*psub2  + Tmod[3];
      t6  = Tmod[6]*psub2  + Tmod[7];
      t10 = Tmod[10]*psub2 + Tmod[11];
      for(int j=jmin_f; j<=jmax_f; j++)
      {
         offset = slice_offset + j*dimf.nx;
         psub1 = (j-nysub2);
         t1 = Tmod[1]*psub1;
         t5 = Tmod[5]*psub1;
         t9 = Tmod[9]*psub1;
         for(int i=imin_f; i<=imax_f; i++)
         {
            v = offset + i;

            if( fmsk[v]>0)
            {
               n++;

               psub0 = (i-nxsub2);

               ptrg0 = Tmod[0]*psub0 + t1 + t2;
               ptrg1 = Tmod[4]*psub0 + t5 + t6;
               ptrg2 = Tmod[8]*psub0 + t9 + t10;

               subject_image_value = sclfim[v];
               target_image_value = linearInterpolator(ptrg0, ptrg1, ptrg2, sclbim, dimb.nx, dimb.ny, dimb.nz, dimb.np);
               sum1 += subject_image_value;
               sum2 += target_image_value;
               sum12 += (subject_image_value*target_image_value);
               sum11 += (subject_image_value*subject_image_value);
               sum22 += (target_image_value*target_image_value);
            }
         }
      }
   }

   for(int k=kmin_b; k<=kmax_b; k++)
   {
      slice_offset = k*dimb.np;
      ptrg2 = (k-nztrg2);
      t2  = invTmod[2]*ptrg2  + invTmod[3];
      t6  = invTmod[6]*ptrg2  + invTmod[7];
      t10 = invTmod[10]*ptrg2 + invTmod[11];
      for(int j=jmin_b; j<=jmax_b; j++)
      {
         offset = slice_offset + j*dimb.nx;
         ptrg1 = (j-nytrg2);
         t1 = invTmod[1]*ptrg1;
         t5 = invTmod[5]*ptrg1;
         t9 = invTmod[9]*ptrg1;
         for(int i=imin_b; i<=imax_b; i++)
         {
            v = offset + i;

            if( bmsk[v]>0)
            {
               n++;

               ptrg0 = (i-nxtrg2);

               psub0 = invTmod[0]*ptrg0 + t1 + t2;
               psub1 = invTmod[4]*ptrg0 + t5 + t6;
               psub2 = invTmod[8]*ptrg0 + t9 + t10;

               subject_image_value = linearInterpolator(psub0, psub1, psub2, sclfim, dimf.nx, dimf.ny, dimf.nz, dimf.np);
               target_image_value = sclbim[v];

               sum1 += subject_image_value;
               sum2 += target_image_value;
               sum12 += (subject_image_value*target_image_value);
               sum11 += (subject_image_value*subject_image_value);
               sum22 += (target_image_value*target_image_value);
            }
         }
      }
   }

   free(invT);

   if( n > 0 )
   {
//      cost = (sum12 - sum1*sum2/n);
      cost = -(sum12 - sum1*sum2/n);  // since it is a minimization problem
      cost /= sqrt( sum11 - sum1*sum1/n ); 
      cost /= sqrt( sum22 - sum2*sum2/n ); 
   }

   return(cost);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// This function computes and returns a 4x4 transformation matrix T.
// Applying this transformation to a point p=(x',y',z') in the image
// coordinates system (ICS) yields the coordinates (x,y,z) of the same point with
// respect to the magnet coordinates system.  That is: (x,y,z)=T(x',y',z').
// In ART, the origin of the ICS is the center of the image
// volume.  The x-axis in the ICS is pointing to the right of the image
// (which is not necessarily the right of the subject!).  The y-axis in the
// ICS is pointing down.  The z-axis is determined by the right-hand-rule.
void transformation_to_magnet_coordinates(nifti_1_header hdr, float *T)
{
   float rowvec[3];
   float columnvec[3];
   float normalvec[3];
   float centervec[3];

   //printf("sizeof_hdr = %d\n", hdr.sizeof_hdr);
   //printf("number of dimensions = %d\n", hdr.dim[0]);
   //printf("matrix size = %d x %d x %d\n", hdr.dim[1], hdr.dim[2], hdr.dim[3]);
   //printf("voxel size = %8.6f x %8.6f x %8.6f\n", hdr.pixdim[1], hdr.pixdim[2], hdr.pixdim[3]);
   //printf("datatype = %d\n",hdr.datatype);
   //printf("vox_offset = %d\n", (int)hdr.vox_offset);
   //printf("qform_code = %d\n", hdr.qform_code);
   //printf("sform_code = %d\n", hdr.sform_code);
   //printf("magic code = %s\n", hdr.magic);
   //printf("srow_x: %f %f %f %f\n", hdr.srow_x[0],hdr.srow_x[1],hdr.srow_x[2],hdr.srow_x[3]);
   //printf("srow_y: %f %f %f %f\n", hdr.srow_y[0],hdr.srow_y[1],hdr.srow_y[2],hdr.srow_y[3]);
   //printf("srow_z: %f %f %f %f\n", hdr.srow_z[0],hdr.srow_z[1],hdr.srow_z[2],hdr.srow_z[3]);
   //printf("quatern_b = %f\n", hdr.quatern_b);
   //printf("quatern_c = %f\n", hdr.quatern_c);
   //printf("quatern_d = %f\n", hdr.quatern_d);
   //printf("qoffset_x = %f\n",hdr.qoffset_x);
   //printf("qoffset_y = %f\n",hdr.qoffset_y);
   //printf("qoffset_z = %f\n",hdr.qoffset_z);

   int nx,ny,nz;
   float dx,dy,dz;

   nx = hdr.dim[1];
   ny = hdr.dim[2];
   nz = hdr.dim[3];

   dx=hdr.pixdim[1];
   dy=hdr.pixdim[2];
   dz=hdr.pixdim[3];

   if(hdr.qform_code>0 )
   {
      float dum;
      mat44 R;
   
      R=nifti_quatern_to_mat44( hdr.quatern_b, hdr.quatern_c, hdr.quatern_d,
      hdr.qoffset_x, hdr.qoffset_y, hdr.qoffset_z, dx, dy, dz, hdr.pixdim[0]);

      dum = R.m[0][0]*R.m[0][0] + R.m[1][0]*R.m[1][0] + R.m[2][0]*R.m[2][0];
      dum = sqrtf(dum);

      if(dum != 0.0)
      {
         // note that the -tive signs converts from NIFTI's RAS to ART's LAI
         rowvec[0]=-R.m[0][0]/dum;
         rowvec[1]=R.m[1][0]/dum;
         rowvec[2]=-R.m[2][0]/dum;
      }
      else
      {
         rowvec[0]=1.0;
         rowvec[1]=0.0;
         rowvec[2]=0.0;
      }

      dum = R.m[0][1]*R.m[0][1] + R.m[1][1]*R.m[1][1] + R.m[2][1]*R.m[2][1];
      dum = sqrtf(dum);

      if(dum != 0.0)
      {
         // note that the -tive signs converts from NIFTI's RAS to ART's LAI
         columnvec[0]=-R.m[0][1]/dum;
         columnvec[1]=R.m[1][1]/dum;
         columnvec[2]=-R.m[2][1]/dum;
      }
      else
      {
         columnvec[0]=0.0;
         columnvec[1]=1.0;
         columnvec[2]=0.0;
      }

      dum = R.m[0][2]*R.m[0][2] + R.m[1][2]*R.m[1][2] + R.m[2][2]*R.m[2][2];
      dum = sqrtf(dum);

      if(dum != 0.0)
      {
         // note that the -tive signs converts from NIFTI's RAS to ART's LAI
         normalvec[0]=-R.m[0][2]/dum;
         normalvec[1]=R.m[1][2]/dum;
         normalvec[2]=-R.m[2][2]/dum;
      }
      else
      {
         normalvec[0]=0.0;
         normalvec[1]=0.0;
         normalvec[2]=1.0;
      }

      // note that the -tive signs converts from NIFTI's RAS to ART's LAI
      centervec[0] = -R.m[0][3] +
      rowvec[0] * dx*(nx-1.0)/2.0 +
      columnvec[0] * dy*(ny-1.0)/2.0;

      centervec[1] = R.m[1][3] +
      rowvec[1] * dx*(nx-1.0)/2.0 +
      columnvec[1] * dy*(ny-1.0)/2.0;

      centervec[2] = -R.m[2][3] +
      rowvec[2] * dx*(nx-1.0)/2.0 +
      columnvec[2] * dy*(ny-1.0)/2.0; 
   }
   else if(hdr.sform_code>0)
   {
      float dum;

      dum = hdr.srow_x[0]*hdr.srow_x[0] + hdr.srow_y[0]*hdr.srow_y[0] + hdr.srow_z[0]*hdr.srow_z[0];
      dum = sqrtf(dum);

      if(dum != 0.0)
      {
         // note that the -tive signs converts from NIFTI's RAS to ART's LAI
         rowvec[0]=-hdr.srow_x[0]/dum;
         rowvec[1]=hdr.srow_y[0]/dum;
         rowvec[2]=-hdr.srow_z[0]/dum;
      }
      else
      {
         rowvec[0]=1.0;
         rowvec[1]=0.0;
         rowvec[2]=0.0;
      }

      dum= hdr.srow_x[1]*hdr.srow_x[1] + hdr.srow_y[1]*hdr.srow_y[1] + hdr.srow_z[1]*hdr.srow_z[1];
      dum = sqrtf(dum);

      if(dum != 0.0)
      {
         // note that the -tive signs converts from NIFTI's RAS to ART's LAI
         columnvec[0]=-hdr.srow_x[1]/dum;
         columnvec[1]=hdr.srow_y[1]/dum;
         columnvec[2]=-hdr.srow_z[1]/dum;
      }
      else
      {
         columnvec[0]=0.0;
         columnvec[1]=1.0;
         columnvec[2]=0.0;
      }

      dum= hdr.srow_x[2]*hdr.srow_x[2] + hdr.srow_y[2]*hdr.srow_y[2] + hdr.srow_z[2]*hdr.srow_z[2];
      dum = sqrtf(dum);

      if(dum != 0.0)
      {
         // note that the -tive signs converts from NIFTI's RAS to ART's LAI
         normalvec[0]=-hdr.srow_x[2]/dum;
         normalvec[1]=hdr.srow_y[2]/dum;
         normalvec[2]=-hdr.srow_z[2]/dum;
      }
      else
      {
         normalvec[0]=0.0;
         normalvec[1]=0.0;
         normalvec[2]=1.0;
      }

      // note that the -tive signs converts from NIFTI's RAS to ART's LAI
      centervec[0] = -hdr.srow_x[3] +
      rowvec[0] * dx*(nx-1.0)/2.0 +
      columnvec[0] * dy*(ny-1.0)/2.0;

      centervec[1] = hdr.srow_y[3] +
      rowvec[1] * dx*(nx-1.0)/2.0 +
      columnvec[1] * dy*(ny-1.0)/2.0;

      centervec[2] = -hdr.srow_z[3] +
      rowvec[2] * dx*(nx-1.0)/2.0 +
      columnvec[2] * dy*(ny-1.0)/2.0; 
   }
   else
   {
      printf("\n**WARNING**: NIFTI header did not contain image orientation information.\n\n");
   }

   //printf("Center Vector = (%7.5lf %7.5lf %7.5lf)\n", centervec[0],centervec[1], centervec[2]);
   //printf("Normal Vector = (%7.5lf %7.5lf %7.5lf)\n", normalvec[0],normalvec[1], normalvec[2]);
   //printf("Row Vector = (%7.5lf %7.5lf %7.5lf)\n", rowvec[0],rowvec[1], rowvec[2]);
   //printf("Column Vector = (%7.5lf %7.5lf %7.5lf)\n", columnvec[0],columnvec[1], columnvec[2]);

   for(int i=0; i<16; i++) T[i]=0.0;

   // The fifteenth element (i.e., element 4,4) is always 1.0.
   T[15]=1.0;
   
   T[0] = rowvec[0];
   T[4] = rowvec[1];
   T[8] = rowvec[2];

   T[1] = columnvec[0];
   T[5] = columnvec[1];
   T[9] = columnvec[2];

   T[2] = normalvec[0];
   T[6] = normalvec[1];
   T[10]= normalvec[2];

   T[3] = centervec[0] + (dz*(nz-1.0)/2.0)*normalvec[0];
   T[7] = centervec[1] + (dz*(nz-1.0)/2.0)*normalvec[1];
   T[11]= centervec[2] + (dz*(nz-1.0)/2.0)*normalvec[2];

   return;
}
//////////////////////////////////////////////////////////////////////////////////////////////////

void atra_to_fsl(float *Matra, float *Mfsl, DIM dimf, DIM trg_dim)
{
   float Tsub[16], Ttrg[16];
   float *inv_Tsub;

   Tsub[0]=1.0;  Tsub[1]=0.0;  Tsub[2]=0.0;  Tsub[3]=(dimf.nx-1.0)*dimf.dx/2.0;
   Tsub[4]=0.0;  Tsub[5]=1.0;  Tsub[6]=0.0;  Tsub[7]=(dimf.ny-1.0)*dimf.dy/2.0;
   Tsub[8]=0.0;  Tsub[9]=0.0;  Tsub[10]=1.0; Tsub[11]=(dimf.nz-1.0)*dimf.dz/2.0;
   Tsub[12]=0.0; Tsub[13]=0.0; Tsub[14]=0.0; Tsub[15]=1.0;

   Ttrg[0]=1.0;  Ttrg[1]=0.0;  Ttrg[2]=0.0;  Ttrg[3]=(trg_dim.nx-1.0)*trg_dim.dx/2.0;
   Ttrg[4]=0.0;  Ttrg[5]=1.0;  Ttrg[6]=0.0;  Ttrg[7]=(trg_dim.ny-1.0)*trg_dim.dy/2.0;
   Ttrg[8]=0.0;  Ttrg[9]=0.0;  Ttrg[10]=1.0; Ttrg[11]=(trg_dim.nz-1.0)*trg_dim.dz/2.0;
   Ttrg[12]=0.0; Ttrg[13]=0.0; Ttrg[14]=0.0; Ttrg[15]=1.0;

   inv_Tsub = inv4(Tsub);

   multi(Ttrg,4,4,Matra,4,4,Mfsl);
   multi(Mfsl,4,4,inv_Tsub,4,4, Mfsl);

   free(inv_Tsub);
}

/////////////////////////////////////////////////
// find sqrt(T) and inverse_sqrt(T)
/////////////////////////////////////////////////
void sqrt_matrix(float *T, float *sqrtT, float *invsqrtT)
{
   float w[3], v[3];
   float theta;

   SE3_to_se3(T, w, v, theta);

   theta /= 2.0;
   se3_to_SE3(sqrtT, w, v, theta);

   theta *= -1.0;
   se3_to_SE3(invsqrtT, w, v, theta);

   return;
}

// bfile: baseline image filename
// ffile: follow-up image filename
void symmetric_registration(SHORTIM &aimpil, const char *bfile, const char *ffile, const char *blmfile,const char *flmfile, int verbose)
{
   char cmnd[1024]="";  // to stores the command to run with system
   short *fmsk, *bmsk;
   float *sclfim, *sclbim;
   short *PILbraincloud;
   DIM PILbraincloud_dim;
   nifti_1_header PILbraincloud_hdr; 

   char filename[1024]="";  // a generic filename for reading/writing stuff

   DIM dimf; // follow-up image dimensions structure
   DIM dimb; // baseline image dimensions structure
   char bprefix[1024]=""; //baseline image prefix
   char fprefix[1024]=""; //follow-up image prefix
   float T[16]; // The unknown transformation matrix that takes points from the follow-up to baseline space 
   float Tf[16]; // The unknown transformation matrix that takes points from the follow-up to mid PIL space 
   float Tb[16]; // The unknown transformation matrix that takes points from the baseline to mid PIL space 
   float Tinter[16]; // Transforms points from the follow-up PIL to baseline PIL spaces
   float *invT;  // inverse of T
   float sqrtTinter[16];
   float invsqrtTinter[16];

   /////////////////////////////////////////////////////////////////////////////////////////////
   // read PILbraincloud.nii from the $ARTHOME directory
   /////////////////////////////////////////////////////////////////////////////////////////////
   sprintf(filename,"%s/PILbrain.nii",ARTHOME);

   PILbraincloud = (short *)read_nifti_image(filename, &PILbraincloud_hdr);

   if(PILbraincloud==NULL)
   {
      printf("Error reading %s, aborting ...\n", filename);
      exit(1);
   }

   if(verbose)
   {
      printf("PIL brain cloud: %s\n",filename);
      printf("PIL brain cloud threshold level = %d\%\n",CLOUD_THRESH);
      printf("PIL brain cloud matrix size = %d x %d x %d (voxels)\n", 
      PILbraincloud_hdr.dim[1], PILbraincloud_hdr.dim[2], PILbraincloud_hdr.dim[3]);
      printf("PIL brain cloud voxel size = %8.6f x %8.6f x %8.6f (mm^3)\n", 
      PILbraincloud_hdr.pixdim[1], PILbraincloud_hdr.pixdim[2], PILbraincloud_hdr.pixdim[3]);
   }

   set_dim(PILbraincloud_dim, PILbraincloud_hdr);
   /////////////////////////////////////////////////////////////////////////////////////////////

   if(verbose)
   {
      printf("Starting unbiased symmetric registration ...\n");
      printf("ARTHOME: %s\n",ARTHOME);
      printf("Maximum number of iterations = %d\n", MAXITER);
      printf("Baseline image: %s\n",bfile);
      printf("Follow-up image: %s\n",ffile);
   }

   // Note: niftiFilename does a few extra checks to ensure that the file has either
   // .hdr or .nii extension, the magic field in the header is set correctly, 
   // the file can be opened and a header can be read.
   if( niftiFilename(bprefix, bfile)==0 )
   {
      exit(0);
   }

   if(verbose)
   {
      printf("Baseline image prefix: %s\n",bprefix);
   }

   if( niftiFilename(fprefix, ffile)==0 )
   {
      exit(0);
   }

   if(verbose)
   {
      printf("Follow-up image prefix: %s\n",fprefix);
   }
   //////////////////////////////////////////////////////////////////////////////////

   float bTPIL[16]; // takes the baseline image to standard PIL orientation 
   float fTPIL[16]; // takes the follow-up image to standard PIL orientation 
   float *ibTPIL; // inverse of bTPIL
   float *ifTPIL; // inverse of fTPIL

   if(verbose) printf("Computing baseline image PIL transformation ...\n");
   if(!opt_newPIL)
      standard_PIL_transformation(bfile, blmfile, verbose, bTPIL);
   else
   {
      new_PIL_transform(bfile, blmfile, bTPIL);
      if(opt_png)
      {
         sprintf(cmnd,"pnmtopng %s_LM.ppm > %s_LM.png",bprefix,bprefix); system(cmnd);
         sprintf(cmnd,"pnmtopng %s_ACPC_axial.ppm > %s_ACPC_axial.png",bprefix,bprefix); system(cmnd);
         sprintf(cmnd,"pnmtopng %s_ACPC_sagittal.ppm > %s_ACPC_sagittal.png",bprefix,bprefix); system(cmnd);
      }
   }

   ibTPIL= inv4(bTPIL);

   if(verbose) printf("Computing follow-up image PIL transformation ...\n");
   if(!opt_newPIL)
      standard_PIL_transformation(ffile, flmfile, verbose, fTPIL);
   else
   {
      new_PIL_transform(ffile, flmfile, fTPIL);
      if(opt_png)
      {
         sprintf(cmnd,"pnmtopng %s_LM.ppm > %s_LM.png",fprefix,fprefix); system(cmnd);
         sprintf(cmnd,"pnmtopng %s_ACPC_axial.ppm > %s_ACPC_axial.png",fprefix,fprefix); system(cmnd);
         sprintf(cmnd,"pnmtopng %s_ACPC_sagittal.ppm > %s_ACPC_sagittal.png",fprefix,fprefix); system(cmnd);
      }
   }

   ifTPIL= inv4(fTPIL);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   // Read baseline and follow-up images
   ///////////////////////////////////////////////////////////////////////////////////////////////
   short *bim; // baseline image
   short *fim; // follow-up image
   nifti_1_header bhdr;  // baseline image NIFTI header
   nifti_1_header fhdr;  // follow-up image NIFTI header

   bim = (short *)read_nifti_image(bfile, &bhdr);

   if(bim==NULL)
   {
      printf("Error reading %s, aborting ...\n", bfile);
      exit(1);
   }

   set_dim(dimb, bhdr);

   fim = (short *)read_nifti_image(ffile, &fhdr);

   if(fim==NULL)
   {
         printf("Error reading %s, aborting ...\n", ffile);
         exit(1);
   }

   set_dim(dimf, fhdr);
   ///////////////////////////////////////////////////////////////////////////////////////////////

   ///////////////////////////////////////////////////////////////////////////////////////////////
   // determine subject and target masks
   ///////////////////////////////////////////////////////////////////////////////////////////////
   {
      float Tdum[16];

      for(int i=0; i<16; i++) Tdum[i]=bTPIL[i];
      bmsk = resliceImage(PILbraincloud, PILbraincloud_dim, dimb, Tdum, LIN);
      for(int v=0; v<dimb.nv; v++) if(bmsk[v]<CLOUD_THRESH) bmsk[v]=0;
      //save_nifti_image("bmsk.nii", bmsk, &bhdr);

      for(int i=0; i<16; i++) Tdum[i]=fTPIL[i];
      fmsk = resliceImage(PILbraincloud, PILbraincloud_dim, dimf, Tdum, LIN);
      for(int v=0; v<dimf.nv; v++) if(fmsk[v]<CLOUD_THRESH) fmsk[v]=0;
      //save_nifti_image("fmsk.nii", fmsk, &fhdr);
      
      delete PILbraincloud;
   }
   ///////////////////////////////////////////////////////////////////////////////////////////////
   
   ///////////////////////////////////////////////////////////////////////////////////////////////
   {
      float bscale;
      float fscale;

      trimExtremes(bim, bmsk, dimb.nv, 0.05);
      trimExtremes(fim, fmsk, dimf.nv, 0.05);

      bscale=imageMean(bim, bmsk, dimb.nv);
      fscale=imageMean(fim, fmsk, dimf.nv);

      sclfim = (float *)calloc(dimf.nv, sizeof(float));
      sclbim = (float *)calloc(dimb.nv, sizeof(float));

      for(int v=0; v<dimf.nv; v++) sclfim[v] = fim[v]/fscale;
      for(int v=0; v<dimb.nv; v++) sclbim[v] = bim[v]/bscale;
   }

#if 0
// for this to work kmin's and kmax's should be defined as global variables and removed as 
// local variables in the cost functions
   ///////////////////////////////////////////////////////////////////////////////////////////////
   // determine mask limits, to limit loop indices in cost_function
   // seems to save 1-2% processing time
   {
      int ii, jj, kk;
      imin_f=dimf.nx; imax_f=0;
      jmin_f=dimf.ny; jmax_f=0;
      kmin_f=dimf.nz; kmax_f=0;
      for(int v=0; v<dimf.nv; v++)
      if( fmsk[v]>0)
      {
         kk = v/dimf.np;
         if( kk<kmin_f) kmin_f=kk;
         if( kk>kmax_f) kmax_f=kk;

         jj = (v%dimf.np)/dimf.nx;
         if( jj<jmin_f) jmin_f=jj;
         if( jj>jmax_f) jmax_f=jj;

         ii = (v%dimf.np)%dimf.nx;
         if( ii<imin_f) imin_f=ii;
         if( ii>imax_f) imax_f=ii;
      }

      imin_b=dimb.nx; imax_b=0;
      jmin_b=dimb.ny; jmax_b=0;
      kmin_b=dimb.nz; kmax_b=0;
      for(int v=0; v<dimb.nv; v++)
      if( bmsk[v]>0)
      {
         kk = v/dimb.np;
         if( kk<kmin_b) kmin_b=kk;
         if( kk>kmax_b) kmax_b=kk;

         jj = (v%dimb.np)/dimb.nx;
         if( jj<jmin_b) jmin_b=jj;
         if( jj>jmax_b) jmax_b=jj;

         ii = (v%dimb.np)%dimb.nx;
         if( ii<imin_b) imin_b=ii;
         if( ii>imax_b) imax_b=ii;
      }
   }
   ///////////////////////////////////////////////////////////////////////////////////////////////
#endif
   
   {
      double relative_change;
      double mincost, oldmincost, cost;
      double (*cost_function)(float *T, DIM dimb, DIM dimf, float *sclbim, float *sclfim,short *bmsk, short *fmsk);
      float P[6];
      float Pmin[6];
      float stepsize[6]={0.25, 0.25, 0.25, 0.1, 0.1, 0.1};  // stepsize used in optimization
      //float iP[6]={3.0, 3.0, 3.0, 1.5, 1.5, 1.5}; // interval used in optimization
      // New interval makes it twice as fast with same resutls
      float iP[6]={1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // interval used in optimization 

      cost_function=ssd_cost_function; // used for T1 to T1 registration
//      cost_function=ncc_cost_function; 
      set_to_I(Tinter,4);

      // initially assume Tinter=Identity matrix
      multi(ibTPIL, 4, 4,  fTPIL, 4,  4, T);
      oldmincost = mincost = cost_function(T, dimb, dimf, sclbim, sclfim,bmsk,fmsk);

      for(int j=0; j<6; j++) P[j]=Pmin[j]=0.0;

      // automatically sets the step size for the first three variables
      //if( dimf.dx<dimb.dx) stepsize[0]=dimf.dx/5.0;
      //else stepsize[0]=dimb.dx/5.0;
      //if( dimf.dy<dimb.dy) stepsize[1]=dimf.dy/5.0;
      //else stepsize[1]=dimb.dy/5.0;
      //if( dimf.dz<dimb.dz) stepsize[2]=dimf.dz/5.0;
      //else stepsize[2]=dimb.dz/5.0;
      //printf("Step sizes = %f %f %f %f %f %f\n",stepsize[0],stepsize[1],stepsize[2],stepsize[3],stepsize[4],stepsize[5]); 

      if(verbose)
      {
         printf("Tolerance = %3.1e\n",TOLERANCE);
         printf("Initial cost = %f\n", mincost);
      }

      for(int iter=1; iter<=MAXITER; iter++)
      {
         if(verbose)
         {
            printf("Iteration %d ...\n",iter);
         }

         for(int i=0; i<6; i++)
         {
            for(P[i] = Pmin[i]-iP[i]; P[i]<=Pmin[i]+iP[i]; P[i]+=stepsize[i] )
            {
               set_transformation(P[0], P[1], P[2], P[3], P[4], P[5], "ZXYT", Tinter);
               multi(Tinter, 4, 4,  fTPIL, 4,  4, T);
               multi(ibTPIL, 4, 4,  T, 4,  4, T);
   
               cost = cost_function(T, dimb, dimf, sclbim, sclfim,bmsk,fmsk);
   
               if( cost < mincost )
               {
                  Pmin[i]=P[i];
                  mincost = cost;
               }
            }
            P[i]=Pmin[i];

            if(verbose)
            {
               printf("P0=%f P1=%f P2=%f P3=%f P4=%f P5=%f\n", Pmin[0], Pmin[1], Pmin[2], Pmin[3], Pmin[4], Pmin[5]);
            }
         }

         if(oldmincost != 0.0)
         {
           relative_change=(oldmincost-mincost)/fabs(oldmincost);
         }
         else
         {
           relative_change=0.0;
         }

         if(verbose)
         {
            printf("Cost = %f\n", mincost);
            printf("Relative change = %3.1e x 100%\n", relative_change );
         }
   
         if( oldmincost==0.0 || relative_change <= TOLERANCE )
            break;
         else
            oldmincost = mincost;
      }

      set_transformation(P[0], P[1], P[2], P[3], P[4], P[5], "ZXYT", Tinter);

      if( Tinter[0]!=1.0 || Tinter[1]!=0.0 || Tinter[2]!=0.0 || Tinter[3]!=0.0 ||
      Tinter[4]!=0.0 || Tinter[5]!=1.0 || Tinter[6]!=0.0 || Tinter[7]!=0.0 ||
      Tinter[8]!=0.0 || Tinter[9]!=0.0 || Tinter[10]!=1.0 || Tinter[11]!=0.0 ||
      Tinter[12]!=0.0 || Tinter[13]!=0.0 || Tinter[14]!=0.0 || Tinter[15]!=1.0)
      {
         // Tinter does not equal identity matrix
         sqrt_matrix(Tinter, sqrtTinter, invsqrtTinter);
      }
      {
         // Tinter equals identity matrix
         for(int i=0; i<16; i++) sqrtTinter[i]=invsqrtTinter[i]=Tinter[i];
      }
      multi(sqrtTinter, 4, 4,  fTPIL, 4,  4, Tf);
      multi(invsqrtTinter, 4, 4,  bTPIL, 4,  4, Tb);
   }
   
   /////////////////////////////////////////////////
   // save transformation matrices
   /////////////////////////////////////////////////
   {
      FILE *fp;

      ////////////////////////////////////////////////////////////////////////////////////////////
      //sprintf(filename,"%s_to_midpoint.mrx",fprefix);
      sprintf(filename,"%s_PIL.mrx",fprefix);
      fp = fopen(filename,"w");
      if(fp != NULL)
      {
         fprintf(fp,"# %s to midpoint rigid-body registration matrix computed by KAIBA",ffile);
         printMatrix(Tf, 4, 4, "", fp);
         fclose(fp);
      }
      else
      {
         printf("Warning: cound not write to %s\n", filename);
      }
      ////////////////////////////////////////////////////////////////////////////////////////////

      ////////////////////////////////////////////////////////////////////////////////////////////
      //sprintf(filename,"%s_to_midpoint.mrx",bprefix);
      sprintf(filename,"%s_PIL.mrx",bprefix);
      fp = fopen(filename,"w");
      if(fp != NULL)
      {
         fprintf(fp,"# %s to midpoint rigid-body registration matrix computed by KAIBA",bfile);
         printMatrix(Tb, 4, 4, "", fp);
         fclose(fp);
      }
      else
      {
         printf("Warning: cound not write to %s\n", filename);
      }
      ////////////////////////////////////////////////////////////////////////////////////////////
      
      free(invT);
   }
   /////////////////////////////////////////////////

   /////////////////////////////////////////////////
   // save registred images
   /////////////////////////////////////////////////
   {
      SHORTIM bimpil; // baseline image after transformation to standard PIL space
      SHORTIM fimpil; // follow-up image after transformation to standard PIL space

      invT = inv4(Tf);
      fimpil.v= resliceImage(fim, dimf, PILbraincloud_dim, invT, LIN);
      set_dim(fimpil, PILbraincloud_dim);
      sprintf(PILbraincloud_hdr.descrip,"Created by ART's KAIBA module");
      sprintf(filename,"%s_PIL.nii",fprefix);
      save_nifti_image(filename, fimpil.v, &PILbraincloud_hdr);
      free(invT);

      invT = inv4(Tb);
      bimpil.v = resliceImage(bim, dimb, PILbraincloud_dim, invT, LIN);
      set_dim(bimpil, PILbraincloud_dim);
      sprintf(PILbraincloud_hdr.descrip,"Created by ART's KAIBA module");
      sprintf(filename,"%s_PIL.nii",bprefix);
      save_nifti_image(filename, bimpil.v, &PILbraincloud_hdr);
      free(invT);

      set_dim(aimpil, PILbraincloud_dim);
      aimpil.v = (short *)calloc(aimpil.nv, sizeof(short));
      for(int i=0; i<aimpil.nv; i++) 
         aimpil.v[i] = (short)( (bimpil.v[i] + fimpil.v[i])/2.0 + 0.5 );

      delete bimpil.v;
      delete fimpil.v;
   }
   /////////////////////////////////////////////////

   delete sclbim;
   delete sclfim;
   delete bmsk;
   delete fmsk;
}

///////////////////////////////////////////////////////////////////////////////////////////////

void compute_lm_transformation(char *lmfile, SHORTIM im, float *A)
{
   FILE *fp;
   int NLM;
   int r;
   int R;
   float *LM; // 4xNLM matrix
   float *CM; // 4xNLM matrix
   int cm[3]; // landmarks center of mass
   int lm[3];

   fp=fopen(lmfile, "r");

   if(fp==NULL) 
   {
      printf("Could not find %s, aborting ...\n",lmfile);
      exit(0);
   }

   fread(&NLM, sizeof(int), 1, fp);
   fread(&r, sizeof(int), 1, fp);
   fread(&R, sizeof(int), 1, fp);
   SPH searchsph(R);
   SPH testsph(r);
   SPH refsph(r);
   LM = (float *)calloc(4*NLM, sizeof(float));
   CM = (float *)calloc(4*NLM, sizeof(float));

   if(opt_v)
   {
      printf("\nLandmark detection ...\n");
      printf("Landmarks file: %s\n", lmfile);
      printf("Number of landmarks sought = %d\n", NLM);
   }

   for(int n=0; n<NLM; n++)
   {
      fread(&cm[0], sizeof(int), 1, fp);
      fread(&cm[1], sizeof(int), 1, fp);
      fread(&cm[2], sizeof(int), 1, fp);
      fread(refsph.v, sizeof(float), refsph.n, fp);

      CM[0*NLM + n]=(cm[0] - (im.nx-1)/2.0)*im.dx; 
      CM[1*NLM + n]=(cm[1] - (im.ny-1)/2.0)*im.dy;
      CM[2*NLM + n]=(cm[2] - (im.nz-1)/2.0)*im.dz;
      CM[3*NLM + n]=1;

      detect_lm(searchsph, testsph, im, cm, refsph, lm);

      LM[0*NLM + n]=(lm[0] - (im.nx-1)/2.0)*im.dx; 
      LM[1*NLM + n]=(lm[1] - (im.ny-1)/2.0)*im.dy;
      LM[2*NLM + n]=(lm[2] - (im.nz-1)/2.0)*im.dz;
      LM[3*NLM + n]=1;
   }

   fclose(fp);

   float *invLMLMT;
   float LMLMT[16];
   float CMLMT[16];

   mat_mat_trans(LM, 4, NLM, LM , 4, LMLMT);
   invLMLMT = inv4(LMLMT);
   mat_mat_trans(CM, 4, NLM, LM , 4, CMLMT);
   multi(CMLMT,4,4,invLMLMT,4,4, A);

   free(LM);
   free(CM);
   free(invLMLMT);
}

///////////////////////////////////////////////////////////////////////////////////////////////

void find_roi(nifti_1_header *subimhdr, SHORTIM pilim, float pilT[],const char *side, const char *prefix)
{
   DIM subdim;

   set_dim(subdim, subimhdr);

   // these are stored in $ARTHOME/<side>.nii file as hdr.dim[5,6,7]
   int imin;
   int jmin;
   int kmin;

   int mskvox;
   int vox;
   short *stndrd_roi;

   char filename[512];
   FILE *fp;

   // hcim is a pre-processed (transformed) version of subim (the original input image in native orientation)
   // readied for hippocampus segmentation.
   SHORTIM hcim; 
  
   // hcim matrix and voxel dimensions are set to a starndard size
   set_dim(hcim, pilim);

   stndrd_roi = (short *)calloc(hcim.nv, sizeof(short));

   // hcT is an affine transformation from subim to hcim
   float hcT[16];

   sprintf(filename,"%s/%s.mdl",ARTHOME,side);
   compute_lm_transformation(filename, pilim, hcT);
   multi(hcT,4,4, pilT, 4,4, hcT);

   //sprintf(filename,"%s_%s.mrx",prefix,side);
   //fp=fopen(filename,"w");
   //printMatrix(hcT,4,4,"",fp);
   //fclose(fp);

   ////////////////////////////////////////////////////////////////////////////////////////////

   SHORTIM msk; 
   nifti_1_header mskhdr;
   
   sprintf(filename,"%s/%s.nii",ARTHOME,side);
   msk.v = (short *)read_nifti_image(filename, &mskhdr);

   if(msk.v==NULL) exit(0);

   msk.nx = mskhdr.dim[1];
   msk.ny = mskhdr.dim[2];
   msk.nz = mskhdr.dim[3];
   msk.np = msk.nx*msk.ny;
   msk.nv = msk.np*msk.nz;
   msk.dx = mskhdr.pixdim[1];
   msk.dy = mskhdr.pixdim[2];
   msk.dz = mskhdr.pixdim[3];
   imin = mskhdr.dim[5];
   jmin = mskhdr.dim[6];
   kmin = mskhdr.dim[7];
   //number of atlases: (mskhdr.dim[4]-1)/2;

   ////////////////////////////////////////////////////////////////////////////////////////////
   {
      short *ntv_spc_roi;
      float T[16];

      if( side[0]=='r')
         sprintf(filename,"%s_RHROI.nii",prefix);
      if( side[0]=='l')
         sprintf(filename,"%s_LHROI.nii",prefix);

      for(int n=0; n<hcim.nv; n++) stndrd_roi[n] = 0;

      for(int k=0; k<msk.nz; k++)
      for(int j=0; j<msk.ny; j++)
      for(int i=0; i<msk.nx; i++)
      {
         stndrd_roi[(k+kmin)*hcim.np + (j+jmin)*hcim.nx + (i+imin)] = msk.v[k*msk.np + j*msk.nx + i];
      }

      for(int i=0; i<16; i++) T[i]=hcT[i];

      ntv_spc_roi = resliceImage(stndrd_roi, hcim.nx, hcim.ny, hcim.nz, hcim.dx, hcim.dy, hcim.dz,
      subdim.nx, subdim.ny, subdim.nz, subdim.dx, subdim.dy, subdim.dz, T, LIN);

      save_nifti_image(filename, ntv_spc_roi, subimhdr);

      free(ntv_spc_roi); free(stndrd_roi);
   }

   return;
}

///////////////////////////////////////////////////////////////////////////////////////////////

void setMX(short *image, short *msk, int nv, int *high, float percent)
{
   short min, max;
   int *histogram;
   int hsize;			/* histogram size */
   int b;
   int i;

   int nmax;
   int n;

   for(int i=0; i<nv; i++) 
   {
      if(msk[i]==0) image[i]=0;
      if(image[i]<0) image[i]=0;
   }

   minmax(image,nv,min,max);

   hsize = max+1;

   histogram=(int *)calloc(hsize,sizeof(int));

   for(int i=0; i<nv; i++)
   {
      b = image[i];

      if(b>=0 && b<hsize) histogram[ b ]++;
   }

   nmax = (int)( percent * (nv-histogram[0])/100.0);

   n=0;
   for(i=0;i<hsize;i++)
   {
      n += histogram[hsize-1-i];
      if(n>nmax) break;
   }
   *high=hsize-1-i; 

   free(histogram);
}

///////////////////////////////////////////////////////////////////////////////////////////////

double compute_hi(char *imfile, char *roifile)
{
   float fuzzy_parenchymasize=0.0;
   int gm_pk_srch_strt;
   int roisize; // number of non-zero voxels in roi
   float fuzzy_roisize=0.0;
   short *roi;
   short *im;
   nifti_1_header hdr;
   int nx, ny, nz, np, nv;
   float dx, dy, dz;
   short roimin, roimax; // minimum and maximum voxels values in the ROI image
   int mx;
   int nbin;

   //if(opt_v)
   //{
   //   printf("Computing HI ...\n");
   //   printf("Image file: %s\n", imfile);
   //   printf("ROI file: %s\n", roifile);
   //}

   roi = (short *)read_nifti_image(roifile, &hdr);
   nx = hdr.dim[1];
   ny = hdr.dim[2];
   nz = hdr.dim[3];
   dx = hdr.pixdim[1];
   dy = hdr.pixdim[2];
   dz = hdr.pixdim[3];
   nv = nx*ny*nz;
   np = nx*ny;

   minmax(roi,nv,roimin,roimax);

   //if(opt_v)
   //{
   //   printf("Matrix size = %d x %d x %d\n", nx, ny, nz);
   //   printf("Voxel size = %f x %f x %f\n", dx, dy, dz);
   //   printf("Min = %d Max = %d\n", roimin, roimax);
   //}

   roisize = 0;
   fuzzy_roisize=0.0;
   for(int i=0; i<nv; i++) 
   {
      if( roi[i]>0 ) roisize++;
   
      fuzzy_roisize += (1.0*roi[i])/roimax;
   }

   //if(opt_v)
   //{
      //printf("ROI size = %d\n", roisize);
      //printf("Fuzzy ROI size = %f\n", fuzzy_roisize);
   //}

   im = (short *)read_nifti_image(imfile, &hdr);
   setMX(im, roi, nv, &mx, HISTCUTOFF);

   //if(opt_v)
   //   printf("MX = %d\n",mx);

   /////////////////////////////////////////////////////////////
   int hist_thresh;
   int im_thresh;
   int im_min, im_max;
   double *hist;
   double *fit;
   double mean[MAXNCLASS+1];
   double var[MAXNCLASS+1];
   double p[MAXNCLASS+1];
   short *label;
   double hmax=0.0;
   int gmpk=0;
   int gmclass;
   double mindiff;
   
   // initialize min and max variables
   for(int i=0; i<nv; i++)
   {
      if( roi[i] > 0)
      {
         im_min=im_max=im[i];
         break;
      }
   }

   // find im_min and im_max amongst the core voxels
   for(int i=0; i<nv; i++)
   {
      if( roi[i] > 0)
      {
         if( im[i]<im_min ) im_min=im[i];
         else if( im[i]>im_max ) im_max=im[i];
      }
   }

   //if(opt_v)
   //   printf("im_min=%d im_max=%d\n",im_min,im_max);

   nbin = im_max-im_min+1;
   hist = (double *)calloc(nbin, sizeof(double));
   fit = (double *)calloc(nbin, sizeof(double));
   label = (short *)calloc(nbin, sizeof(short));

   // initialize hist to 0 (to be sure)
   for(int i=0; i<nbin; i++) hist[i]=0.0;

   // set hist of the core voxels
   for(int i=0; i<nv; i++)
   {
      if( roi[i] > 0)
      {
         hist[im[i]-im_min]++;
      }
   }

   for(int i=0; i<nbin; i++) hist[i]/=roisize;

   gm_pk_srch_strt = (mx * MXFRAC - im_min);
   if( gm_pk_srch_strt < 0) gm_pk_srch_strt=0;

   //if(opt_v)
   //   printf("gm_pk_srch_strt = %d\n",gm_pk_srch_strt);

   int nclass=5;

   EMFIT1d(hist, fit, label, nbin, mean, var, p, nclass, 1000);

   for(int i=gm_pk_srch_strt; i<nbin; i++)
   {
         if( fit[i] > hmax ) 
         { 
            hmax=fit[i]; 
            gmpk=i; 
         }
   }

   gmclass=0;
   mindiff = abs( mean[0] - gmpk );

   for(int i=0; i<nclass; i++)
   {
         //if(opt_v) printf("class=%d mean=%lf p=%lf\n", i, mean[i], p[i] );
         if( abs( mean[i] - gmpk ) < mindiff )
         {
            mindiff = abs( mean[i] - gmpk );
            gmclass=i;
         }
   }

   //if(!opt_v)
//      for(int i=0; i<nbin; i++) printf("%d %lf %lf %d\n",i, hist[i], fit[i], label[i]);

   double csfvol=0.0;

   // Al's method for find the hist_threshold
   hist_thresh = (int)(gmpk - mx*MXFRAC2 + 0.5);

//printf("\nhist_thresh=%d\n",hist_thresh);
//printf("\ngmpk=%d\n",gmpk);

   for(int i=0; i<hist_thresh; i++)
   {
      csfvol += hist[i];
   }

   im_thresh = hist_thresh + im_min;
   fuzzy_parenchymasize= 0.0;
   for(int v=0; v<nv; v++)
   {
      if( im[v] >= im_thresh && roi[v]>0 ) fuzzy_parenchymasize += (1.0*roi[v])/roimax;
   }

   //if(opt_v)
      //printf("\n** HI=%lf gmpk=%d Thresh=%d **\n\n", 1.0-csfvol, gmpk, hist_thresh);
   //printf("%s %s %lf %f\n", imfile,roifile, 1.0-csfvol, fuzzy_parenchymasize/fuzzy_roisize);

   free(hist); free(fit); free(label);
   /////////////////////////////////////////////////////////////

   return(1.0-csfvol);
}

///////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
   char cmnd[1024]=""; // stores the command to run with system
   opt_ppm=YES;
   opt_txt=NO;

   float hi;
   FILE *fp;
   char filename[1024]="";  // a generic filename for reading/writing stuff

   char opprefix[512]=""; // prefix used for reading/writing output files

   opt_txt=NO; // avoids saving *ACPC.txt files

   char bprefix[1024]=""; //baseline image prefix
   char fprefix[1024]=""; //follow-up image prefix

   char roifile[1024]="";

   char blmfile[1024]="";
   char flmfile[1024]="";

   char bfile[1024]=""; // baseline image filename
   char ffile[1024]=""; // follow-up image filename

   if(argc==1) print_help_and_exit();

   while ((opt = getoption(argc, argv, options)) != -1 )
   {
      switch (opt) 
      {
         case 'V':
            printf("KAIBA Version 2.0 released March 1, 2016.\n");
            printf("Author: Babak A. Ardekani, Ph.D.\n");
            exit(0);
         case 'v':
            opt_v=YES;
            break;
         case 'g':
            opt_png=YES;
            break;
         case 'n':
            opt_newPIL=NO;
            break;
         case 'p':
            sprintf(opprefix,"%s",optarg);
            break;
         case 'l':
            sprintf(blmfile,"%s",optarg);
            break;
         case 'm':
            sprintf(flmfile,"%s",optarg);
            break;
         case 'b':
            sprintf(bfile,"%s",optarg);
            break;
         case 'f':
            sprintf(ffile,"%s",optarg);
            break;
         case '?':
            print_help_and_exit();
      }
   }

   getARTHOME();

   // Ensure that an output prefix has been specified at the command line.
   if( opprefix[0]=='\0' )
   {
      printf("Please specify an output prefix using -p argument.\n");
      exit(0);
   }

   //////////////////////////////////////////////////////////////////////////////////
   // Receive input image filenames and deteremine their prefix
   //////////////////////////////////////////////////////////////////////////////////

   // Ensure that a baseline image has been specified at the command line.
   if( bfile[0]=='\0' )
   {
      printf("Please specify a baseline image using -b argument.\n");
      exit(0);
   }

   float pilT[16];

   /////////////////////////////////////////////////////////////////////////////////////////////
   // read PILbraincloud.nii from the $ARTHOME directory
   // The only reason this is done is to read dimensions of PILbrain.nii
   /////////////////////////////////////////////////////////////////////////////////////////////
   short *PILbraincloud;
   DIM PILbraincloud_dim;
   nifti_1_header PILbraincloud_hdr; 

   sprintf(filename,"%s/PILbrain.nii",ARTHOME);

   PILbraincloud = (short *)read_nifti_image(filename, &PILbraincloud_hdr);

   if(PILbraincloud==NULL)
   {
         printf("Error reading %s, aborting ...\n", filename);
         exit(1);
   }

   set_dim(PILbraincloud_dim, PILbraincloud_hdr);
   delete PILbraincloud;
   /////////////////////////////////////////////////////////////////////////////////////////////
      
   sprintf(filename,"%s.csv",opprefix);
   fp = fopen(filename,"w");
   if(fp==NULL) file_open_error(filename);
   fprintf(fp,"image, roi, hi\n");

   // for longitudinal case
   if( bfile[0]!='\0' && ffile[0]!='\0')
   {
      SHORTIM aimpil; // average of baseline and follow-up images after transformation to standard PIL space

      if( niftiFilename(bprefix, bfile)==0 ) exit(0);
      if( niftiFilename(fprefix, ffile)==0 ) exit(0);

      symmetric_registration(aimpil, bfile, ffile, blmfile, flmfile, opt_v);

      ///////////////////////////////////////////////////////////////////////////////////////////////
      // processing baseline image
      ///////////////////////////////////////////////////////////////////////////////////////////////
      SHORTIM bim; // baseline image
      nifti_1_header bim_hdr;  // baseline image NIFTI header

      bim.v = (short *)read_nifti_image(bfile, &bim_hdr);

      if(bim.v==NULL)
      {
         printf("Error reading %s, aborting ...\n", bfile);
         exit(1);
      }

      sprintf(filename,"%s_PIL.mrx",bprefix);
      loadTransformation(filename, pilT);

      find_roi(&bim_hdr, aimpil, pilT, "lhc3", bprefix);
      find_roi(&bim_hdr, aimpil, pilT, "rhc3", bprefix);

      free(bim.v);

      sprintf(roifile,"%s_RHROI.nii",bprefix);
      hi=compute_hi(bfile, roifile);
      fprintf(fp,"%s, %s, %lf\n",bfile,roifile,hi);

      sprintf(roifile,"%s_LHROI.nii",bprefix);
      hi=compute_hi(bfile, roifile);
      fprintf(fp,"%s, %s, %lf\n",bfile,roifile,hi);
      ///////////////////////////////////////////////////////////////////////////////////////////////

      ///////////////////////////////////////////////////////////////////////////////////////////////
      // processing followup image
      ///////////////////////////////////////////////////////////////////////////////////////////////
      SHORTIM fim; // followup image
      nifti_1_header fim_hdr;  // followup image NIFTI header

      fim.v = (short *)read_nifti_image(ffile, &fim_hdr);

      if(fim.v==NULL)
      {
         printf("Error reading %s, aborting ...\n", ffile);
         exit(1);
      }

      sprintf(filename,"%s_PIL.mrx",fprefix);
      loadTransformation(filename, pilT);

      find_roi(&fim_hdr, aimpil, pilT, "lhc3", fprefix);
      find_roi(&fim_hdr, aimpil, pilT, "rhc3", fprefix);

      free(fim.v);

      sprintf(roifile,"%s_RHROI.nii",fprefix);
      hi=compute_hi(ffile, roifile);
      fprintf(fp,"%s, %s, %lf\n",ffile,roifile,hi);

      sprintf(roifile,"%s_LHROI.nii",fprefix);
      hi=compute_hi(ffile, roifile);
      fprintf(fp,"%s, %s, %lf\n",ffile,roifile,hi);
      ///////////////////////////////////////////////////////////////////////////////////////////////

      delete aimpil.v;
   }
   else // for cross-sectional case
   {
      SHORTIM bimpil; // baseline image after transformation to standard PIL space
      // Note: niftiFilename does a few extra checks to ensure that the file has either
      // .hdr or .nii extension, the magic field in the header is set correctly, 
      // the file can be opened and a header can be read.
      if( niftiFilename(bprefix, bfile)==0 ) exit(0);

      if(opt_v) printf("Baseline image prefix: %s\n",bprefix);

      ///////////////////////////////////////////////////////////////////////////////////////////////
      // Read baseline image
      ///////////////////////////////////////////////////////////////////////////////////////////////
      SHORTIM bim; // baseline image
      nifti_1_header bhdr;  // baseline image NIFTI header
      DIM dimb; // baseline image dimensions structure

      bim.v = (short *)read_nifti_image(bfile, &bhdr);

      if(bim.v==NULL)
      {
         printf("Error reading %s, aborting ...\n", bfile);
         exit(1);
      }

      set_dim(dimb, bhdr);
      set_dim(bim, dimb);
      ///////////////////////////////////////////////////////////////////////////////////////////////
      
   
      float bTPIL[16]; // takes the baseline image to standard PIL orientation 
      float *invT;
      if(opt_v) printf("Computing baseline image PIL transformation ...\n");
      if(!opt_newPIL)
         standard_PIL_transformation(bfile, blmfile, opt_v, bTPIL);
      else
      {
         new_PIL_transform(bfile,blmfile,bTPIL);
         if(opt_png)
         {
            sprintf(cmnd,"pnmtopng %s_LM.ppm > %s_LM.png",bprefix,bprefix); system(cmnd);
            sprintf(cmnd,"pnmtopng %s_ACPC_axial.ppm > %s_ACPC_axial.png",bprefix,bprefix); system(cmnd);
            sprintf(cmnd,"pnmtopng %s_ACPC_sagittal.ppm > %s_ACPC_sagittal.png",bprefix,bprefix); system(cmnd);
         }
      }

      invT = inv4(bTPIL);
      bimpil.v = resliceImage(bim.v, dimb, PILbraincloud_dim, invT, LIN);
      set_dim(bimpil, PILbraincloud_dim);
      free(invT);

      sprintf(PILbraincloud_hdr.descrip,"Created by ART's KAIBA module");
      sprintf(filename,"%s_PIL.nii",bprefix);
      save_nifti_image(filename, bimpil.v, &PILbraincloud_hdr);

      find_roi(&bhdr, bimpil, bTPIL, "lhc3", bprefix);
      find_roi(&bhdr, bimpil, bTPIL, "rhc3", bprefix);

      delete bimpil.v;

      sprintf(roifile,"%s_RHROI.nii",bprefix);
      hi=compute_hi(bfile, roifile);
      fprintf(fp,"%s, %s, %lf\n",bfile,roifile,hi);

      sprintf(roifile,"%s_LHROI.nii",bprefix);
      hi=compute_hi(bfile, roifile);
      fprintf(fp,"%s, %s, %lf\n",bfile,roifile,hi);
   }
   fclose(fp);
}
