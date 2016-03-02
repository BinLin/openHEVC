/*
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * Frame multithreading support functions
 * @see doc/multithreading.txt
 */

#include "config.h"

#include <stdint.h>

#if HAVE_PTHREADS
#include <pthread.h>
#elif HAVE_W32THREADS
#include "compat/w32pthreads.h"
#elif HAVE_OS2THREADS
#include "compat/os2threads.h"
#endif

#include "avcodec.h"
#include "internal.h"
#include "pthread_internal.h"
#include "thread.h"

#include "libavutil/avassert.h"
#include "libavutil/buffer.h"
#include "libavutil/common.h"
#include "libavutil/cpu.h"
#include "libavutil/frame.h"
#include "libavutil/internal.h"
#include "libavutil/log.h"
#include "libavutil/mem.h"
#include "libavcodec/hevc.h"

#define MAX_POC      1024
/**
 * Context used by codec threads and stored in their AVCodecInternal thread_ctx_frame.
 */
typedef struct PerThreadContext {
    struct FrameThreadContext *parent;  //!< 指向线程池数据结构

    pthread_t      thread;          //!< 线程ID
    int            thread_init;     //!< thread 是否init
    pthread_cond_t input_cond;      ///< Used to wait for a new packet from the main thread.主线程通知相应的解码线程,数据包已经准备好,解码线程可以开始解码
    pthread_cond_t progress_cond;   ///< Used by child threads to wait for progress to change.用于通知线程状态改变
    pthread_cond_t output_cond;     ///< Used by the main thread to wait for frames to finish.用于解码线程通知主线程输出已经准备好

    pthread_mutex_t mutex;          ///< Mutex used to protect the contents of the PerThreadContext.
    pthread_mutex_t progress_mutex; ///< Mutex used to protect frame progress values and progress_cond.

    AVCodecContext *avctx;          ///< Context used to decode packets passed to this thread.

    AVPacket       avpkt;           ///< Input packet (for decoding) or output (for encoding).

    AVFrame *frame;                 ///< Output frame (for decoding) or input (for encoding).存储解码后的数据(解码器)
    int     got_frame;              ///< The output of got_picture_ptr from the last avcodec_decode_video() call.
    int     result;                 ///< The result of the last codec decode/encode() call.

    enum {
        STATE_INPUT_READY,          ///< Set when the thread is awaiting a packet.工作线程等待主线程分配数据包
        STATE_SETTING_UP,           ///< Set before the codec has called ff_thread_finish_setup().主线程给工作线程分配好数据包,工作线程解码
        STATE_GET_BUFFER,           /**<
                                     * Set when the codec calls get_buffer().
                                     * State is returned to STATE_SETTING_UP afterwards.
                                     */
        STATE_GET_FORMAT,           /**<
                                     * Set when the codec calls get_format().
                                     * State is returned to STATE_SETTING_UP afterwards.
                                     */
        STATE_SETUP_FINISHED        ///< Set after the codec has called ff_thread_finish_setup().工作线程完成解码,通知主线程可以输出
    } state;   //!< 初创建的线程的state为默认值STATE_INPUT_READY

    /**
     * Array of frames passed to ff_thread_release_buffer().
     * Frames are released after all threads referencing them are finished.
     */
    AVFrame *released_buffers;
    int  num_released_buffers;
    int      released_buffers_allocated;

    AVFrame *requested_frame;       ///< AVFrame the codec passed to get_buffer()
    int      requested_flags;       ///< flags passed to get_buffer() for requested_frame

    const enum AVPixelFormat *available_formats; ///< Format array for get_format()
    enum AVPixelFormat result_format;            ///< get_format() result
} PerThreadContext;

/**
 * Context stored in the client AVCodecInternal thread_ctx_frame.
 */
typedef struct FrameThreadContext {
    PerThreadContext *threads;     ///< The contexts for each thread.链表保存指向线程池的每个线程
    PerThreadContext *prev_thread; ///< The last thread submit_packet() was called on.

    pthread_mutex_t buffer_mutex;  ///< Mutex used to protect get/release_buffer().互斥锁

    int next_decoding;             ///< The next context to submit a packet to.
    int next_finished;             ///< The next context to return output from.

    int delaying;                  /**<
                                    * Set for the first N packets, where N is the number of threads.
                                    * While it is set, ff_thread_en/decode_frame won't return any results.
                                    */

    int die;                       ///< Set when threads should exit.
    int is_decoded[MAX_POC];
    int last_Tid;
    void* frames_ref[MAX_POC];
    void* frames_data[MAX_POC];
    pthread_mutex_t il_progress_mutex; ///< Mutex used to protect frame progress values and progress_cond.
    pthread_cond_t  il_progress_cond;   ///< Used by child threads to wait for progress to change.
} FrameThreadContext;

#if FF_API_GET_BUFFER
#define THREAD_SAFE_CALLBACKS(avctx) \
((avctx)->thread_safe_callbacks || (!(avctx)->get_buffer && (avctx)->get_buffer2 == avcodec_default_get_buffer2))
#else
#define THREAD_SAFE_CALLBACKS(avctx) \
((avctx)->thread_safe_callbacks || (avctx)->get_buffer2 == avcodec_default_get_buffer2)
#endif

/**
 * Codec worker thread.
 *
 * Automatically calls ff_thread_finish_setup() if the codec does
 * not provide an update_thread_context method, or if the codec returns
 * before calling it.
 */
static attribute_align_arg void *frame_worker_thread(void *arg)
{
    PerThreadContext *p = arg;
    FrameThreadContext *fctx = p->parent;
    AVCodecContext *avctx = p->avctx;
    const AVCodec *codec = avctx->codec;

    pthread_mutex_lock(&p->mutex);
    while (1) {  //!< p->state的初始默认值为STATE_INPUT_READY
            while (p->state == STATE_INPUT_READY && !fctx->die)  //!< 等待用户线程给当前线程提交数据包
                pthread_cond_wait(&p->input_cond, &p->mutex);

        if (fctx->die) break;

        if (!codec->update_thread_context && THREAD_SAFE_CALLBACKS(avctx))  //!< 判断是否可以帧级并行
            ff_thread_finish_setup(avctx);   //!< 更改当前工作线程的状态,通知主线程可以让下一个线程开始工作解码

        av_frame_unref(p->frame);
        p->got_frame = 0;
        p->result = codec->decode(avctx, p->frame, &p->got_frame, &p->avpkt); //!< 解码过程

        if ((p->result < 0 || !p->got_frame) && p->frame->buf[0]) {
            if (avctx->internal->allocate_progress)
                av_log(avctx, AV_LOG_ERROR, "A frame threaded decoder did not "
                       "free the frame on failure. This is a bug, please report it.\n");
            av_frame_unref(p->frame);
        }

        if (p->state == STATE_SETTING_UP) ff_thread_finish_setup(avctx);

        pthread_mutex_lock(&p->progress_mutex);
#if 0 //BUFREF-FIXME
        for (i = 0; i < MAX_BUFFERS; i++)
            if (p->progress_used[i] && (p->got_frame || p->result<0 || avctx->codec_id != AV_CODEC_ID_H264)) {
                p->progress[i][0] = INT_MAX;
                p->progress[i][1] = INT_MAX;
            }
#endif
        p->state = STATE_INPUT_READY;

        pthread_cond_broadcast(&p->progress_cond);  //!< 通知等待线程,该线程状态已经改变
        pthread_cond_signal(&p->output_cond);       //!< 通知主线程,该线程输出已经准备好
        pthread_mutex_unlock(&p->progress_mutex);
    }
    pthread_mutex_unlock(&p->mutex);

    return NULL;
}

/**
 * Update the next thread's AVCodecContext with values from the reference thread's context.
 *
 * @param dst The destination context.
 * @param src The source context.
 * @param for_user 0 if the destination is a codec thread, 1 if the destination is the user's thread
 */
static int update_context_from_thread(AVCodecContext *dst, AVCodecContext *src, int for_user)
{
    int err = 0;

    if (dst != src) {
        dst->time_base = src->time_base;
        dst->width     = src->width;
        dst->height    = src->height;
        dst->pix_fmt   = src->pix_fmt;

        dst->coded_width  = src->coded_width;
        dst->coded_height = src->coded_height;

        dst->has_b_frames = src->has_b_frames;
        dst->idct_algo    = src->idct_algo;

        dst->bits_per_coded_sample = src->bits_per_coded_sample;
        dst->sample_aspect_ratio   = src->sample_aspect_ratio;
        dst->dtg_active_format     = src->dtg_active_format;

        dst->profile = src->profile;
        dst->level   = src->level;

        dst->bits_per_raw_sample = src->bits_per_raw_sample;
        dst->ticks_per_frame     = src->ticks_per_frame;
        dst->color_primaries     = src->color_primaries;

        dst->color_trc   = src->color_trc;
        dst->colorspace  = src->colorspace;
        dst->color_range = src->color_range;
        dst->chroma_sample_location = src->chroma_sample_location;

        dst->hwaccel = src->hwaccel;
        dst->hwaccel_context = src->hwaccel_context;

        dst->channels       = src->channels;
        dst->sample_rate    = src->sample_rate;
        dst->sample_fmt     = src->sample_fmt;
        dst->channel_layout = src->channel_layout;
        dst->internal->hwaccel_priv_data = src->internal->hwaccel_priv_data;
    }

    if (for_user) {
        dst->delay       = src->thread_count - 1;
        dst->coded_frame = src->coded_frame;
    } else {
        if (dst->codec->update_thread_context)
            err = dst->codec->update_thread_context(dst, src);
    }

    return err;
}

/**
 * Update the next thread's AVCodecContext with values set by the user.
 *
 * @param dst The destination context.
 * @param src The source context.
 * @return 0 on success, negative error code on failure
 */
static int update_context_from_user(AVCodecContext *dst, AVCodecContext *src)
{
#define copy_fields(s, e) memcpy(&dst->s, &src->s, (char*)&dst->e - (char*)&dst->s);
    dst->flags          = src->flags;
    dst->quality_id     = src->quality_id;
    dst->draw_horiz_band= src->draw_horiz_band;
    dst->get_buffer2    = src->get_buffer2;
#if FF_API_GET_BUFFER
FF_DISABLE_DEPRECATION_WARNINGS
    dst->get_buffer     = src->get_buffer;
    dst->release_buffer = src->release_buffer;
FF_ENABLE_DEPRECATION_WARNINGS
#endif

    dst->opaque   = src->opaque;
    dst->debug    = src->debug;
    dst->debug_mv = src->debug_mv;

    dst->slice_flags = src->slice_flags;
    dst->flags2      = src->flags2;

    copy_fields(skip_loop_filter, subtitle_header);

    dst->frame_number     = src->frame_number;
    dst->reordered_opaque = src->reordered_opaque;
    dst->thread_safe_callbacks = src->thread_safe_callbacks;

    if (src->slice_count && src->slice_offset) {
        if (dst->slice_count < src->slice_count) {
            int err = av_reallocp_array(&dst->slice_offset, src->slice_count,
                                        sizeof(*dst->slice_offset));
            if (err < 0)
                return err;
        }
        memcpy(dst->slice_offset, src->slice_offset,
               src->slice_count * sizeof(*dst->slice_offset));
    }
    dst->slice_count = src->slice_count;
    return 0;
#undef copy_fields
}

/// Releases the buffers that this decoding thread was the last user of.
static void release_delayed_buffers(PerThreadContext *p)
{
    FrameThreadContext *fctx = p->parent;

    while (p->num_released_buffers > 0) {
        AVFrame *f;

        pthread_mutex_lock(&fctx->buffer_mutex);

        // fix extended data in case the caller screwed it up
        av_assert0(p->avctx->codec_type == AVMEDIA_TYPE_VIDEO ||
                   p->avctx->codec_type == AVMEDIA_TYPE_AUDIO);
        f = &p->released_buffers[--p->num_released_buffers];
        f->extended_data = f->data;
        av_frame_unref(f);

        pthread_mutex_unlock(&fctx->buffer_mutex);
    }
}

static int submit_packet(PerThreadContext *p, AVPacket *avpkt)
{
    FrameThreadContext *fctx = p->parent;
    PerThreadContext *prev_thread = fctx->prev_thread;  //!< 上一个接受数据包的线程
    const AVCodec *codec = p->avctx->codec;

    if (!avpkt->size && !(codec->capabilities & CODEC_CAP_DELAY)) return 0;

    pthread_mutex_lock(&p->mutex);

    release_delayed_buffers(p);  //!< 释放与当前线程相关的内存

    if (prev_thread) {
        int err;
        if (prev_thread->state == STATE_SETTING_UP) {  //!< 需要等待前一个线程状态转为STATE_SETUP_FINISHED,现进行线程等待
            pthread_mutex_lock(&prev_thread->progress_mutex);  //!< 互斥量保护条件
            while (prev_thread->state == STATE_SETTING_UP)
                pthread_cond_wait(&prev_thread->progress_cond, &prev_thread->progress_mutex);
            pthread_mutex_unlock(&prev_thread->progress_mutex);
        }

        err = update_context_from_thread(p->avctx, prev_thread->avctx, 0); //!< 如果之前的线程状态变为 STATE_SETUP_FINISHED, 更新相关内容到当前线程
        if (err) {
            pthread_mutex_unlock(&p->mutex);
            return err;
        }
    }

    av_packet_unref(&p->avpkt); //!< 释放与p->avpkt相关联的动态分配的内存,并重新初始化
    av_packet_ref(&p->avpkt, avpkt);  //!< 拷贝解码数据,以及相关参数

    p->state = STATE_SETTING_UP;
    //!< pthread_cond_signal(条件) 将唤醒等待该条件的某个线程,这里是线程p
    pthread_cond_signal(&p->input_cond);//!< 发送码流输入准备好的信号给解码线程(pthread_cond_waite(&p->input_cond),让相应的解码线程解码
    pthread_mutex_unlock(&p->mutex);

    /*
     * If the client doesn't have a thread-safe get_buffer(),
     * then decoding threads call back to the main thread,
     * and it calls back to the client here.
     */

FF_DISABLE_DEPRECATION_WARNINGS
    if (!p->avctx->thread_safe_callbacks && (
         p->avctx->get_format != avcodec_default_get_format ||
#if FF_API_GET_BUFFER
         p->avctx->get_buffer ||
#endif
         p->avctx->get_buffer2 != avcodec_default_get_buffer2)) {
FF_ENABLE_DEPRECATION_WARNINGS
        while (p->state != STATE_SETUP_FINISHED && p->state != STATE_INPUT_READY) {
            int call_done = 1;
            pthread_mutex_lock(&p->progress_mutex);
            while (p->state == STATE_SETTING_UP)
                pthread_cond_wait(&p->progress_cond, &p->progress_mutex);

            switch (p->state) {
            case STATE_GET_BUFFER:
                p->result = ff_get_buffer(p->avctx, p->requested_frame, p->requested_flags);
                break;
            case STATE_GET_FORMAT:
                p->result_format = ff_get_format(p->avctx, p->available_formats);
                break;
            default:
                call_done = 0;
                break;
            }
            if (call_done) {
                p->state  = STATE_SETTING_UP;
                pthread_cond_signal(&p->progress_cond);
            }
            pthread_mutex_unlock(&p->progress_mutex);
        }
    }

    fctx->prev_thread = p;
    fctx->next_decoding++;

    return 0;
}
//!< 帧级解码函数
int ff_thread_decode_frame(AVCodecContext *avctx,
                           AVFrame *picture, int *got_picture_ptr,
                           AVPacket *avpkt)
{
    FrameThreadContext *fctx = avctx->internal->thread_ctx_frame;
    int finished = fctx->next_finished;  //!< 下一个返回解码帧的线程序号
    PerThreadContext *p;  //!< 编解码使用的线程,这些线程存储在 avctx->internal->thread_ctx_frame
    int err;

    /*
     * Submit a packet to the next decoding thread.
     */

    p = &fctx->threads[fctx->next_decoding];  //!< 等待数据包的线程
    err = update_context_from_user(p->avctx, avctx);  //!< 将用户提交的avctx复制到解码线程p->avctx里
    if (err) return err;
    err = submit_packet(p, avpkt);  //!< 将数据码流交给对应的解码线程,来实现线程状态的改变
    if (err) return err;

    /*
     * If we're still receiving the initial packets, don't return a frame.
     * 若线程池中的线程还没用完,则继续接收数据包,不输出
     */

    if (fctx->next_decoding > (avctx->thread_count_frame-1-(avctx->codec_id == AV_CODEC_ID_FFV1)))
        fctx->delaying = 0;

    if (fctx->delaying) {
        *got_picture_ptr=0;
        if (avpkt->size)
            return avpkt->size;
    }

    /*
     * Return the next available frame from the oldest thread.
     * If we're at the end of the stream, then we have to skip threads that
     * didn't output a frame, because we don't want to accidentally signal
     * EOF (avpkt->size == 0 && *got_picture_ptr == 0).
     */
    finished = fctx->next_decoding >= avctx->thread_count_frame ? 0:fctx->next_decoding;
    do {
        p = &fctx->threads[finished++];

        if (p->state != STATE_INPUT_READY) { //!< 当前线程的数据状态不是STATE_INPUT_READY,等待该线程解码完数据包
            pthread_mutex_lock(&p->progress_mutex);   //!< 上锁
            while (p->state != STATE_INPUT_READY)
                pthread_cond_wait(&p->output_cond, &p->progress_mutex);  //!< 通知主线程,解码数据包已经准备好输出
            pthread_mutex_unlock(&p->progress_mutex);  //!< 解锁
        }

        av_frame_move_ref(picture, p->frame);   //!< outPut_frame
        *got_picture_ptr = p->got_frame;
        picture->pkt_dts = p->avpkt.dts;

        /*
         * A later call with avkpt->size == 0 may loop over all threads,
         * including this one, searching for a frame to return before being
         * stopped by the "finished != fctx->next_finished" condition.
         * Make sure we don't mistakenly return the same frame again.
         */
        p->got_frame = 0;

        if (finished >= avctx->thread_count_frame) finished = 0;
    } while (!avpkt->size && !*got_picture_ptr && finished != fctx->next_finished);

    update_context_from_thread(avctx, p->avctx, 1);  //!< 将线程提交的avctx复制到用户avctx里

    if (fctx->next_decoding >= avctx->thread_count_frame) fctx->next_decoding = 0;

    fctx->next_finished = finished;

    /* return the size of the consumed packet if no error occurred */
    return (p->result >= 0) ? avpkt->size : p->result;
}

void ff_thread_report_progress(ThreadFrame *f, int n, int field)
{
    PerThreadContext *p;
    volatile int *progress = f->progress ? (int*)f->progress->data : NULL;

    if (!progress || progress[field] >= n) return;

    p = f->owner->internal->thread_ctx_frame;

    if (f->owner->debug&FF_DEBUG_THREADS)
        av_log(f->owner, AV_LOG_DEBUG, "%p finished %d field %d\n", progress, n, field);

    pthread_mutex_lock(&p->progress_mutex);
    progress[field] = n;
    pthread_cond_broadcast(&p->progress_cond);  //@! 以广播的方式,唤醒所有等待该条件的所有线程
    pthread_mutex_unlock(&p->progress_mutex);
}

void ff_thread_await_progress(ThreadFrame *f, int n, int field)
{
    PerThreadContext *p;
    volatile int *progress = f->progress ? (int*)f->progress->data : NULL;

    if (!progress || progress[field] >= n) return;

    p = f->owner->internal->thread_ctx_frame;

    if (f->owner->debug&FF_DEBUG_THREADS)
        av_log(f->owner, AV_LOG_DEBUG, "thread awaiting %d field %d from %p\n", n, field, progress);

    pthread_mutex_lock(&p->progress_mutex);
    while (progress[field] < n)
        pthread_cond_wait(&p->progress_cond, &p->progress_mutex);
    pthread_mutex_unlock(&p->progress_mutex);
}

#ifdef SVC_EXTENSION
void ff_thread_report_il_progress(AVCodecContext *avxt, int poc, void * in_ref, void *in_data) {
/*
    - Called by the  lower layer decoder to report that the frame used as reference at upper layers
      is either decoded or allocated in the frame-based.
    - Set the status to 1.
    - This operation is signaled at the parent the frame-based thread.
*/
    PerThreadContext *p;
    FrameThreadContext *fctx;
    p = avxt->internal->thread_ctx_frame;
    fctx = p->parent;

    poc = poc & (MAX_POC-1);
    if (avxt->debug&FF_DEBUG_THREADS)
        av_log(avxt, AV_LOG_DEBUG, "ff_thread_report_il_progress %d\n", poc);
    pthread_mutex_lock(&fctx->il_progress_mutex);
    if(fctx->is_decoded[poc] == 3 && in_ref) {
        ff_hevc_unref_frame(avxt->priv_data, in_ref, ~0);
        fctx->is_decoded[poc] = 0;
    } else {
            fctx->is_decoded[poc]     = 1;
            fctx->frames_ref[poc]     = in_ref;
            fctx->frames_data[poc]    = in_data;
            pthread_cond_broadcast(&fctx->il_progress_cond);
    }
    
    pthread_mutex_unlock(&fctx->il_progress_mutex);
}

int ff_thread_get_il_up_status(AVCodecContext *avxt, int poc)
{
    /*
     - Get the status of the lower layer picture used as reference for inter-layer prediction.
     */
    int res;
    PerThreadContext *p;
    FrameThreadContext *fctx;
    p = avxt->internal->thread_ctx_frame;
    fctx = p->parent;
    poc = poc & (MAX_POC-1);
    if (avxt->debug&FF_DEBUG_THREADS)
        av_log(avxt, AV_LOG_DEBUG, "ff_thread_get_il_up_status %d \n", poc);
    pthread_mutex_lock(&fctx->il_progress_mutex);
    res = fctx->is_decoded[poc];
    pthread_mutex_unlock(&fctx->il_progress_mutex);
    return res;
}

void ff_thread_await_il_progress(AVCodecContext *avxt, int poc, void ** out) {
    /*
     - Wait untill the lower layer picture used for inter-layer reference picture is either allocated or decoded
     - The condition is that the $is_decoded$ variable of the corresponding poc is diffetent from 0.
     - $copy_opaque$ allows to access to the $parent$ variable of the lower layer decoder.
     - Get the reference of the reference picture picture from lower layer decoder.
     
     */
    FrameThreadContext *fctx = ((AVCodecContext *)avxt->BL_avcontext)->internal->thread_ctx_frame;
    poc = poc & (MAX_POC-1);
    if (avxt->debug&FF_DEBUG_THREADS)
        av_log(avxt, AV_LOG_DEBUG, "ff_thread_await_il_progress %d \n", poc);
    pthread_mutex_lock(&fctx->il_progress_mutex);
    while(fctx->is_decoded[poc] == 0)
        pthread_cond_wait(&fctx->il_progress_cond, &fctx->il_progress_mutex);
    *out = fctx->frames_data[poc];
    pthread_mutex_unlock(&fctx->il_progress_mutex);
}

void ff_thread_report_il_status(AVCodecContext *avxt, int poc, int status) {
    /*
     - Called by the upper layer decoder to report that the picture using this reference frame is decoded and the lower layer is not any more required by upper layer decoder
     */
    FrameThreadContext *fctx = ((AVCodecContext *)avxt->BL_avcontext)->internal->thread_ctx_frame;
    AVCodecContext *avxt_bl = (AVCodecContext *)avxt->BL_avcontext;
    poc = poc & (MAX_POC-1);
    if (avxt->debug&FF_DEBUG_THREADS)
        av_log(avxt, AV_LOG_DEBUG, "ff_thread_report_il_status poc %d status %d\n", poc, status);
    pthread_mutex_lock(&fctx->il_progress_mutex);
    if(fctx->is_decoded[poc]==1 ) {
        if(fctx->frames_ref[poc])
            ff_hevc_unref_frame(avxt_bl->priv_data, fctx->frames_ref[poc], ~0);
        fctx->is_decoded[poc] = 0;
    } else
        fctx->is_decoded[poc] = 3;
    fctx->frames_data[poc] = NULL;
    fctx->frames_ref[poc] = NULL;
    pthread_mutex_unlock(&fctx->il_progress_mutex);
}

void ff_thread_report_il_status2(AVCodecContext *avxt, int poc, int status) {
    /*
    - Called by the lower layer decoder to report the new status of the picture as removed
     
    */
    PerThreadContext *p;
    FrameThreadContext *fctx;
    p = avxt->internal->thread_ctx_frame;
    fctx = p->parent;
    poc = poc & (MAX_POC-1);
    if (avxt->debug&FF_DEBUG_THREADS)
        av_log(avxt, AV_LOG_DEBUG, "ff_thread_report_il_status2\n");
    pthread_mutex_lock(&fctx->il_progress_mutex);
    fctx->is_decoded[poc] = status;
    if(!status) {
        fctx->frames_data[poc] = NULL;
                fctx->frames_ref[poc] = NULL;
    }
    pthread_mutex_unlock(&fctx->il_progress_mutex);
}
#endif

void ff_thread_finish_setup(AVCodecContext *avctx) {
    PerThreadContext *p = avctx->internal->thread_ctx_frame;

    if (!(avctx->active_thread_type&FF_THREAD_FRAME)) return;  //!< 不支持frame并行,返回

    if(p->state == STATE_SETUP_FINISHED){
        av_log(avctx, AV_LOG_WARNING, "Multiple ff_thread_finish_setup() calls\n");
    }

    pthread_mutex_lock(&p->progress_mutex);
    p->state = STATE_SETUP_FINISHED;
    pthread_cond_broadcast(&p->progress_cond);
    pthread_mutex_unlock(&p->progress_mutex);
}

/// Waits for all threads to finish.
static void park_frame_worker_threads(FrameThreadContext *fctx, int thread_count)
{
    int i;

    for (i = 0; i < thread_count; i++) {
        PerThreadContext *p = &fctx->threads[i];

        if (p->state != STATE_INPUT_READY) {
            pthread_mutex_lock(&p->progress_mutex);
            while (p->state != STATE_INPUT_READY)
                pthread_cond_wait(&p->output_cond, &p->progress_mutex);
            pthread_mutex_unlock(&p->progress_mutex);
        }
        p->got_frame = 0;
    }
}

void ff_frame_thread_free(AVCodecContext *avctx, int thread_count)
{
    FrameThreadContext *fctx = avctx->internal->thread_ctx_frame;
    const AVCodec *codec = avctx->codec;
    int i;

    park_frame_worker_threads(fctx, thread_count); //!< 等待所有线程工作完成,接下来才可以释放

    if (fctx->prev_thread && fctx->prev_thread != fctx->threads)
        if (update_context_from_thread(fctx->threads->avctx, fctx->prev_thread->avctx, 0) < 0) {
            av_log(avctx, AV_LOG_ERROR, "Final thread update failed\n");
            fctx->prev_thread->avctx->internal->is_copy = fctx->threads->avctx->internal->is_copy;
            fctx->threads->avctx->internal->is_copy = 1;
        }

    fctx->die = 1;

    for (i = 0; i < thread_count; i++) {
        PerThreadContext *p = &fctx->threads[i];
        if (p->avctx->active_thread_type&FF_THREAD_SLICE)
            ff_slice_thread_free(p->avctx);
        pthread_mutex_lock(&p->mutex);
        pthread_cond_signal(&p->input_cond);
        pthread_mutex_unlock(&p->mutex);

        if (p->thread_init)
            pthread_join(p->thread, NULL);
        p->thread_init=0;

        if (codec->close)
            codec->close(p->avctx);

        avctx->codec = NULL;

        release_delayed_buffers(p);
        av_frame_free(&p->frame);
    }

    for (i = 0; i < thread_count; i++) {
        PerThreadContext *p = &fctx->threads[i];

        pthread_mutex_destroy(&p->mutex);
        pthread_mutex_destroy(&p->progress_mutex);
        pthread_cond_destroy(&p->input_cond);
        pthread_cond_destroy(&p->progress_cond);
        pthread_cond_destroy(&p->output_cond);
        av_packet_unref(&p->avpkt);
        av_freep(&p->released_buffers);

        if (i) {
            av_freep(&p->avctx->priv_data);
            av_freep(&p->avctx->slice_offset);
        }

        av_freep(&p->avctx->internal);
        av_freep(&p->avctx);
    }

    av_freep(&fctx->threads);
    pthread_mutex_destroy(&fctx->buffer_mutex);
    av_freep(&avctx->internal->thread_ctx_frame);
}

int ff_frame_thread_init(AVCodecContext *avctx)
{
    int thread_count = avctx->thread_count_frame;
    const AVCodec *codec = avctx->codec;
    AVCodecContext *src = avctx;
    FrameThreadContext *fctx;
    int i, err = 0;

#if HAVE_W32THREADS
    w32thread_init();
#endif

    if (!thread_count) {
        int nb_cpus = av_cpu_count();
        if ((avctx->debug & (FF_DEBUG_VIS_QP | FF_DEBUG_VIS_MB_TYPE)) || avctx->debug_mv)
            nb_cpus = 1;
        // use number of cores + 1 as thread count if there is more than one
        if (nb_cpus > 1) //!< CPU个数大于1时,设置线程个数 = nb_cpu + 1
            thread_count = avctx->thread_count_frame = FFMIN(nb_cpus + 1, MAX_AUTO_THREADS);
        else
            thread_count = avctx->thread_count_frame = 1;
    }

    if (thread_count <= 1) {
        avctx->active_thread_type = 0;
        return 0;
    }
    //!< 线程池数据结构
    avctx->internal->thread_ctx_frame = fctx = av_mallocz(sizeof(FrameThreadContext));
    //!< 给线程池的线程分配内存
    fctx->threads = av_mallocz(sizeof(PerThreadContext) * thread_count);
    pthread_mutex_init(&fctx->buffer_mutex, NULL);
    pthread_cond_init(&fctx->il_progress_cond, NULL);
    pthread_mutex_init(&fctx->il_progress_mutex, NULL);
    fctx->delaying = 1;
    //!< 线程池内的线程依次初始化
    for (i = 0; i < thread_count; i++) {
        AVCodecContext *copy = av_malloc(sizeof(AVCodecContext));
        PerThreadContext *p  = &fctx->threads[i];

        pthread_mutex_init(&p->mutex, NULL);
        pthread_mutex_init(&p->progress_mutex, NULL);
        pthread_cond_init(&p->input_cond, NULL);
        pthread_cond_init(&p->progress_cond, NULL);
        pthread_cond_init(&p->output_cond, NULL);

        p->frame = av_frame_alloc();
        if (!p->frame) {
            av_freep(&copy);
            err = AVERROR(ENOMEM);
            goto error;
        }

        p->parent = fctx;
        p->avctx  = copy;

        if (!copy) {
            err = AVERROR(ENOMEM);
            goto error;
        }

        *copy = *src;

        copy->internal = av_malloc(sizeof(AVCodecInternal));
        if (!copy->internal) {
            err = AVERROR(ENOMEM);
            goto error;
        }
        *copy->internal = *src->internal;
        copy->internal->thread_ctx_frame = p;
        copy->internal->pkt = &p->avpkt;

        if (avctx->active_thread_type&FF_THREAD_SLICE)  //!< SLice 级别并行初始化
            ff_slice_thread_init(copy);

        if (!i) {  //!< 初始化线程池的第一个线程的priv_data指向用户线程的priv_data
            src = copy;

            if (codec->init)
                err = codec->init(copy);

            update_context_from_thread(avctx, copy, 1);//!< 更新到主线程的解码上下文
        } else {   //!< 将主线程的priv_data复制给线程池除第一个线程之外的其余线程的priv_data
            copy->priv_data = av_malloc(codec->priv_data_size);
            if (!copy->priv_data) {
                err = AVERROR(ENOMEM);
                goto error;
            }
            memcpy(copy->priv_data, src->priv_data, codec->priv_data_size);
            copy->internal->is_copy = 1;

            if (codec->init_thread_copy)
                err = codec->init_thread_copy(copy);
        }

        if (err) goto error;

        err = AVERROR(pthread_create(&p->thread, NULL, frame_worker_thread, p));
        p->thread_init= !err;
        if(!p->thread_init)
            goto error;
    }

    return 0;

error:
    ff_frame_thread_free(avctx, i+1);

    return err;
}

void ff_thread_flush(AVCodecContext *avctx)
{
    int i;
    FrameThreadContext *fctx = avctx->internal->thread_ctx_frame;

    if (!fctx) return;

    park_frame_worker_threads(fctx, avctx->thread_count_frame);
    if (fctx->prev_thread) {
        if (fctx->prev_thread != &fctx->threads[0])
            update_context_from_thread(fctx->threads[0].avctx, fctx->prev_thread->avctx, 0);
    }

    fctx->next_decoding = fctx->next_finished = 0;
    fctx->delaying = 1;
    fctx->prev_thread = NULL;
    for (i = 0; i < avctx->thread_count_frame; i++) {
        PerThreadContext *p = &fctx->threads[i];
        // Make sure decode flush calls with size=0 won't return old frames
        p->got_frame = 0;
        av_frame_unref(p->frame);

        release_delayed_buffers(p);

        if (avctx->codec->flush)
            avctx->codec->flush(p->avctx);
    }
}

int ff_thread_can_start_frame(AVCodecContext *avctx)
{
    PerThreadContext *p = avctx->internal->thread_ctx_frame;
    if ((avctx->active_thread_type&FF_THREAD_FRAME) && p->state != STATE_SETTING_UP &&
        (avctx->codec->update_thread_context || !THREAD_SAFE_CALLBACKS(avctx))) {
        return 0;
    }
    return 1;
}

static int thread_get_buffer_internal(AVCodecContext *avctx, ThreadFrame *f, int flags)
{
    PerThreadContext *p = avctx->internal->thread_ctx_frame;
    int err;

    f->owner = avctx;

    ff_init_buffer_info(avctx, f->f);

    if (!(avctx->active_thread_type & FF_THREAD_FRAME))
        return ff_get_buffer(avctx, f->f, flags);

    if (p->state != STATE_SETTING_UP &&
        (avctx->codec->update_thread_context || !THREAD_SAFE_CALLBACKS(avctx))) {
        av_log(avctx, AV_LOG_ERROR, "get_buffer() cannot be called after ff_thread_finish_setup()\n");
        return -1;
    }

    if (avctx->internal->allocate_progress) {
        int *progress;
        f->progress = av_buffer_alloc(2 * sizeof(int));
        if (!f->progress) {
            return AVERROR(ENOMEM);
        }
        progress = (int*)f->progress->data;

        progress[0] = progress[1] = -1;
    }

    pthread_mutex_lock(&p->parent->buffer_mutex);
FF_DISABLE_DEPRECATION_WARNINGS
    if (avctx->thread_safe_callbacks || (
#if FF_API_GET_BUFFER
        !avctx->get_buffer &&
#endif
        avctx->get_buffer2 == avcodec_default_get_buffer2)) {
FF_ENABLE_DEPRECATION_WARNINGS
        err = ff_get_buffer(avctx, f->f, flags);
    } else {
        pthread_mutex_lock(&p->progress_mutex);
        p->requested_frame = f->f;
        p->requested_flags = flags;
        p->state = STATE_GET_BUFFER;
        pthread_cond_broadcast(&p->progress_cond);

        while (p->state != STATE_SETTING_UP)
            pthread_cond_wait(&p->progress_cond, &p->progress_mutex);

        err = p->result;

        pthread_mutex_unlock(&p->progress_mutex);

    }
    if (!THREAD_SAFE_CALLBACKS(avctx) && !avctx->codec->update_thread_context)
        ff_thread_finish_setup(avctx);

    if (err)
        av_buffer_unref(&f->progress);

    pthread_mutex_unlock(&p->parent->buffer_mutex);

    return err;
}

enum AVPixelFormat ff_thread_get_format(AVCodecContext *avctx, const enum AVPixelFormat *fmt)
{
    enum AVPixelFormat res;
    PerThreadContext *p = avctx->internal->thread_ctx_frame;
    if (!(avctx->active_thread_type & FF_THREAD_FRAME) || avctx->thread_safe_callbacks ||
        avctx->get_format == avcodec_default_get_format)
        return ff_get_format(avctx, fmt);
    if (p->state != STATE_SETTING_UP) {
        av_log(avctx, AV_LOG_ERROR, "get_format() cannot be called after ff_thread_finish_setup()\n");
        return -1;
    }
    pthread_mutex_lock(&p->progress_mutex);
    p->available_formats = fmt;
    p->state = STATE_GET_FORMAT;
    pthread_cond_broadcast(&p->progress_cond);

    while (p->state != STATE_SETTING_UP)
        pthread_cond_wait(&p->progress_cond, &p->progress_mutex);

    res = p->result_format;

    pthread_mutex_unlock(&p->progress_mutex);

    return res;
}

int ff_thread_get_buffer(AVCodecContext *avctx, ThreadFrame *f, int flags)
{
    int ret = thread_get_buffer_internal(avctx, f, flags);
    if (ret < 0)
        av_log(avctx, AV_LOG_ERROR, "thread_get_buffer() failed\n");
    return ret;
}

void ff_thread_release_buffer(AVCodecContext *avctx, ThreadFrame *f)
{
    PerThreadContext *p = avctx->internal->thread_ctx_frame;
    FrameThreadContext *fctx;
    AVFrame *dst, *tmp;
FF_DISABLE_DEPRECATION_WARNINGS
    int can_direct_free = !(avctx->active_thread_type & FF_THREAD_FRAME) ||
                          avctx->thread_safe_callbacks                   ||
                          (
#if FF_API_GET_BUFFER
                           !avctx->get_buffer &&
#endif
                           avctx->get_buffer2 == avcodec_default_get_buffer2);
FF_ENABLE_DEPRECATION_WARNINGS

    if (!f->f || !f->f->buf[0])
        return;

    if (avctx->debug & FF_DEBUG_BUFFERS)
        av_log(avctx, AV_LOG_DEBUG, "thread_release_buffer called on pic %p\n", f);

    av_buffer_unref(&f->progress);
    f->owner    = NULL;

    if (can_direct_free) {
        av_frame_unref(f->f);
        return;
    }

    fctx = p->parent;
    pthread_mutex_lock(&fctx->buffer_mutex);

    if (p->num_released_buffers + 1 >= INT_MAX / sizeof(*p->released_buffers))
        goto fail;
    tmp = av_fast_realloc(p->released_buffers, &p->released_buffers_allocated,
                          (p->num_released_buffers + 1) *
                          sizeof(*p->released_buffers));
    if (!tmp)
        goto fail;
    p->released_buffers = tmp;

    dst = &p->released_buffers[p->num_released_buffers];
    av_frame_move_ref(dst, f->f);

    p->num_released_buffers++;

fail:
    pthread_mutex_unlock(&fctx->buffer_mutex);
}
