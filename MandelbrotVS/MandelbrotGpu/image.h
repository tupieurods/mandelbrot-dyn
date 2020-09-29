#pragma once

#include <png.h>
#include <cstdio>
#include <cstdlib>
#include "conf.h"

void dwell_color(int* r, int* g, int* b, int dwell)
{
  // black for the Mandelbrot set
  if(dwell >= MAX_DWELL)
  {
    *r = *g = *b = 0;
  }
  else
  {
    // cut at zero
    if(dwell < 0)
      dwell = 0;
    if(dwell <= CUT_DWELL)
    {
      // from black to blue the first half
      *r = *g = 0;
      *b = 128 + dwell * 127 / (CUT_DWELL);
    }
    else
    {
      // from blue to white for the second half
      *b = 255;
      *r = *g = (dwell - CUT_DWELL) * 255 / (MAX_DWELL - CUT_DWELL);
    }
  }
} // dwell_color

/** save the dwell into a PNG file
    @remarks: code to save PNG file taken from here
      (error handling is removed):
    http://www.labbookpages.co.uk/software/imgProc/libPNG.html
 */
void save_image(const char* filename, int* dwells, int w, int h)
{
  char title[] = "Title";
  char text[] = "Mandelbrot set, per-pixel";


  png_bytep row;

  FILE* fp = fopen(filename, "wb");
  png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  png_infop info_ptr = png_create_info_struct(png_ptr);
  // exception handling
  setjmp(png_jmpbuf(png_ptr));
  png_init_io(png_ptr, fp);
  // write header (8 bit colour depth)
  png_set_IHDR(png_ptr, info_ptr, w, h,
    8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
    PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
  // set title
  png_text title_text;
  title_text.compression = PNG_TEXT_COMPRESSION_NONE;
  title_text.key = title;
  title_text.text = text;
  png_set_text(png_ptr, info_ptr, &title_text, 1);
  png_write_info(png_ptr, info_ptr);

  // write image data
  row = static_cast<png_bytep>(malloc(3 * w * sizeof(png_byte)));
  for(int y = 0; y < h; y++)
  {
    for(int x = 0; x < w; x++)
    {
      int r, g, b;
      dwell_color(&r, &g, &b, dwells[y * w + x]);
      row[3 * x + 0] = static_cast<png_byte>(r);
      row[3 * x + 1] = static_cast<png_byte>(g);
      row[3 * x + 2] = static_cast<png_byte>(b);
    }
    png_write_row(png_ptr, row);
  }
  png_write_end(png_ptr, nullptr);

  fclose(fp);
  png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
  png_destroy_write_struct(&png_ptr, static_cast<png_infopp>(nullptr));
  free(row);
} // save_image