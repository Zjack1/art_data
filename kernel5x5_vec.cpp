#include <stdlib.h>
#include <string.h>
#include <vec-c.h>


#define STRIDE_1 1
#define STRIDE_2 2
#define STRIDE_4 4

#define PATTERN_NUM 7
#define PARALLEL_KERNEL 4//并行内核

#define MAX(a,b)    (((a) > (b)) ? (a) : (b))

#define WIDTH 31
#define HEIGHT 31
#define CHANNEL 96

#define KERNEL_WIDTH 5
#define KERNEL_HEIGHT 5
#define KERNELSIZE_ROUND 112

#define WIDTH_OUT (WIDTH-KERNEL_WIDTH+1)  //27
#define HEIGHT_OUT (HEIGHT-KERNEL_HEIGHT+1) //27

#define STRIDE_SHORT MAX(34,(((WIDTH_OUT+29)>>5)<<5)+2) //34//和bank冲突相关，因为dsp中有vpld或者后面的vpst的加载和存储指令，这个和vld还要vst不一样，前者可以跨区间存储，后者都是存储连续的，不过要用int就是32bit来存储，因为一个bank宽是32bit

short data[WIDTH * HEIGHT * CHANNEL]; //27x27 padded to 31x31

short data_out[STRIDE_SHORT * HEIGHT_OUT * PARALLEL_KERNEL];//34 * 27 * 4

short coeff[KERNELSIZE_ROUND * CHANNEL];  //112* 96

short data_unit[WIDTH * HEIGHT];  //31*31

//short coeff_unit[KERNELSIZE_ROUND] = {
//	1, 4, 6, 4, 1, 4, 6, 4, 1, 4, 6, 4, 1, 4, 6, 4,
//	1, 24, 16, 4, 1, 24, 16, 4, 1, 24, 16, 4, 1, 24, 16, 4,
//	4, 16, 6, 24, 4, 16, 6, 24, 4, 16, 6, 24, 4, 16, 6, 24,
//	36, 24, 6, 4, 36, 24, 6, 4, 36, 24, 6, 4, 36, 24, 6, 4,
//	4, 16, 24, 16, 4, 16, 24, 16, 4, 16, 24, 16, 4, 16, 24, 16,
//	1, 4, 6, 4, 1, 4, 6, 4, 1, 4, 6, 4, 1, 4, 6, 4,
//	1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};

short coeff_unit[KERNELSIZE_ROUND] = {
	1, 4, 6, 4, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
	1, 24, 16, 4, 1, 24, 16, 4, 1, 24, 16, 4, 1, 24, 16, 4,
	4, 16, 6, 24, 4, 16, 6, 24, 4, 16, 6, 24, 4, 16, 6, 24,
	36, 24, 6, 4, 36, 24, 6, 4, 36, 24, 6, 4, 36, 24, 6, 4,
	4, 16, 24, 16, 4, 16, 24, 16, 4, 16, 24, 16, 4, 16, 24, 16,
	1, 4, 6, 4, 1, 4, 6, 4, 1, 4, 6, 4, 1, 4, 6, 4,
	1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 };
int main(){

	//init data
	for (int i = 0; i < WIDTH * HEIGHT; i++)//得到一个channel的map
	{
		int row = i / WIDTH;
		int col = i % WIDTH;
		data_unit[row * WIDTH + col] = i;
	}

	
	short* data_ptr = data;//31*31*96
	short* coeff_ptr = coeff; //112* 96
	for (int c = 0; c < CHANNEL; c++)//把map和卷积核都变成96的channel
	{
		memcpy(coeff_ptr, coeff_unit, KERNELSIZE_ROUND * sizeof(short));//void *memcpy(void *str1, const void *str2, size_t n) 从存储区 str2 复制 n 个字符到存储区 str1。
		memcpy(data_ptr, data_unit, WIDTH*HEIGHT * sizeof(short));
		coeff_ptr += KERNELSIZE_ROUND;//卷积核的指针偏移112个，来存储下一个coeff_unit
		data_ptr += WIDTH * HEIGHT;//map的指针偏移31*31，来存储下一个data_unit
	}


	//compute
	data_ptr = data;//指针初始化，指向第一个数
	coeff_ptr = coeff;//指针初始化，指向第一个数

	int data_stride = WIDTH;
	int data_out_stride = STRIDE_SHORT;
	int data_map_stride = WIDTH * HEIGHT;
	int coeff_stride = 16;//short16
	int coeff_map_stride = KERNELSIZE_ROUND;

	unsigned int config_reg0[PATTERN_NUM] = {//7
		0xf0000080,//0 偏移0位
		0x17000180,//1 偏移一位00010111
		0xcc000080,
		0x71000180,
		0xf0000080,
		0xf0000080,
		0x80000480};

	unsigned int config_reg1[PATTERN_NUM] = {
		0xf0000280,//2 偏移两位
		0x17000380,
		0xcc000280,
		0x71000380,
		0xf0000280,
		0xf0000280,
		0x80000680 };


	

	int stride_data[6] = {0*data_stride,1*data_stride,2*data_stride,3*data_stride,4*data_stride,4*data_stride};//data（map）数据指针的偏移
	int stride_coeff[6] = { 16, 32, 48, 64, 80,96 };//卷积核指针的偏移

	//int vst_offset[8] = { (STRIDE_SHORT * HEIGHT_OUT * 0)>>1, (STRIDE_SHORT * HEIGHT_OUT * 1)>>1, (STRIDE_SHORT * HEIGHT_OUT * 2)>>1, (STRIDE_SHORT * HEIGHT_OUT * 3)>>1, 0, 0, 0, 0 };
	int vst_offset[8] = { 0, 5, 10, 30, 0, 0, 0, 0 };

	short* data_x_ptr = data;//将data数组的首地址复制给data_x_ptr
	short* out_x_ptr = data_out;
	

	for (int i = 0; i < WIDTH_OUT; i+=PARALLEL_KERNEL)
	{
		short* data_y_ptr = data_x_ptr;//初始化data指针
		short* out_y_ptr = out_x_ptr;//初始化out指针



		for (int j = 0; j < HEIGHT_OUT; j++)//31*31的图形，5*5的卷积核滑动27次
		{
			data_ptr = data_y_ptr;//初始化data指针
			int8 vacc0 = (int8)0;
			int8 vacc1 = (int8)0;
			//uint16 v11 = (uint16)0;

			for (int p = 0; p < PATTERN_NUM; p++)//7种模板
			{

				for (int c = 0; c < CHANNEL; c++) //96个channel，每次循环回来，读取特征图下一个channel的特征图，循环96次将4个kernel读入的共16个weights，
				{	                              //那么对后得到8个点，每个kernel：2个（起始加滑动一次）
				
					short16 vdata0 = *(short16*)data_ptr; //拿第data一行16个值
					short16 vdata1 = *(short16*)(data_ptr + data_stride);//map内跨度31到第二行拿16个值

					data_ptr += data_map_stride;//指针指向下一个channel的map

					short16 vcoeff = *(short16*)coeff_ptr; //拿卷积核的第一行16个值
					coeff_ptr += coeff_map_stride;//同样指向4个kernel的下一个channel的weights，实际中不是只有4个kernel的，可以将kernel通过4来划分，
					//v11 = vsubsquare(vdata0, vcoeff);
					vacc0 = vsspmac(accumulate, vdata0, vdata1, vcoeff, config_reg0[p], STRIDE_1, vacc0);
					vacc1 = vsspmac(accumulate, vdata0, vdata1, vcoeff, config_reg1[p], STRIDE_1, vacc1);

				}
				data_ptr = data_x_ptr + stride_data[p];//map 指针的偏移下一行
				coeff_ptr = coeff + stride_coeff[p];//p=0时时1 4 6 4共4个kernel，然后指针偏移16到p=1时为1 24 16 4 共4个kernel的 后面类推weights在generate那边先排好了，（卷积核指针的偏移 下一行）
													//针对5x5的是4个kernel排一次，多余的可以除以4来划tile，每次处理完4个再处理下面，直到将num=k的kernel全部处理完
													
			}//循环之后，就完成了前4个kernel完整的5x5卷积，并移动了4次，也就是每个kernel卷积第一行4个点
			short16 vzero16 = short16(0);
			short8 vzero8 = short8(0);

			short8 vresult0 = vsspmac(vzero16, vzero16, vzero16, 0x00000008, STRIDE_1, vacc0);//此处移位为了将32位改为16bit，此处8是任意，实际代码中使用fraction小数bit位，来做移位
			short8 vresult1 = vsspmac(vzero16, vzero16, vzero16, 0x00000008, STRIDE_1, vacc1);

			short16 vtemp0 = vpack(vresult0, vzero8);
			short16 vtemp1 = vpack(vresult1, vzero8);

			int8 a = *(int8*)vst_offset;
			vpst((int8)vtemp0, (int*)out_y_ptr, *(int8*)vst_offset,(char)0x07);
			vpst((int8)vtemp1, (int*)out_y_ptr+1, *(int8*)vst_offset, (char)0x0f);


			data_y_ptr += data_stride; //从第二行开始做channel的卷积
			out_y_ptr += data_out_stride;//输出层指针移位
			coeff_ptr = coeff;//卷积核指针复位

		}
		data_x_ptr += PARALLEL_KERNEL;//指针偏移到第一行的第五列，然后再计算vacc0(前四个每个stride为1)
		out_x_ptr  += PARALLEL_KERNEL;//输出层指针移位
		coeff_ptr = coeff;//卷积核指针复位

	}


	return 0;

}
