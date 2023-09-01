sd - event : fix theoretical leak on error in event_make_signal_data
platform / x86 : dell - wmi : check for kmalloc () errors
ARM : 6076 / 1 : SA1100 : add processor check to sa1110 - cpufreq driver
basic : use for () loop instead of while ()
Staging : Fix leak in drivers / staging / at76_usb . c
rgw : Error check on return of read_line ()
irqchip / gic - v3 : Fix out - of - bound access in gic_set_affinity
[ media ] tda10071 : fix returned symbol rate calculation
calendarspec : free memory when parsing fails (# 4890 )
core bugfix : small memory leak in external command parser
Made it so cores or threads can be allocated on
pgsql : Avoid epoll_ctl () errors when closing connection .
pty : Fix byte loss bug when connecting to pty
kcm : fix a null pointer dereference in kcm_sendmsg ()
mtd : onenand : omap2 : Remove skip initial unlocking support
buf : don ' t realloc for every buf_ensure
[ PATCH ] ieee1394 : fix cross_bound check for null ISO packets
kvm - irqchip : only commit route when irqchip is used
lib - smtp : smtp_server_connection_pending_command_data () - don ' t crash if smtp_parser is NULL
lib - imap - client : Don ' t start another DNS lookup if there ' s already one ongoing .
[ Bluetooth ] Revert session reference counting fix
TG3 : limit reaches - 1
avformat / dss : Use av_freep () to avoid leaving stale pointers in memory
lib - storage : mailbox_list_delete_trash () - Don ' t ignore errors
Input : gpio_keys - fix a memory leak
UDF : Close small mem leak in udf_find_entry ()
diff : avoid potential null pointer dereference on error
fsdev : fix virtfs - proxy - helper cwd
PM : Free memory bitmaps if opening / dev / snapshot fails
mime : properly unbind mime structure in curl_mime_free ().
sulogin : use EXIT_ *
ACPICA : Fix for Linux source translation
fstab - generator : fix trivial leak
cmd : Use ' void ' instead of empty parameter list
[ PATCH ] knfsd : Protect update to sn_nrthreads with lock_kernel
sd - event : fix theoretical leak on error in event_make_signal_data
Protected against garbage in the CapRTL encoder
hls : fix leaking avio_opts on hls_read_header error
modules / usrloc : init _ul_sruid also in mi_child_init
Protected against garbage in the CapRTL encoder
[ Bluetooth ] Revert session reference counting fix
avformat / lrcdec : Fix memory leak in lrc_read_header ()
UDF : Close small mem leak in udf_find_entry ()
[ media ] coda : v4l2 - compliance fix : zero pixel format priv field
rgw / rgw_rest . cc : fix uninitialized static struct
Bugfix : Free shares discarded before beginning submission
Hardware : Fix possible recursive locking in hwregs . c
ARM : zynq : initialize slcr mapping earlier
ARM : zynq : initialize slcr mapping earlier
ahci : fix buffer overrun on invalid state load
slirp : check len against dhcp options array end
hw / openrisc : Avoid using uninitialised variable ' entry '
spapr : fix leak in h_client_architecture_support ()
sparc / leon3 . c : fix memory leak
qcow2 : Fix header extension size check
qdev - monitor : Fix crash when device_add is called with abstract driver
Input : amijoy - add missing platform check
lkdtm : do not leak free page on kmalloc failure
mei : start disconnect request timer consistently
kvm - irqchip : only commit route when irqchip is used
bus : properly unset default bus pointer when destroying last reference
kvm - irqchip : only commit route when irqchip is used
kvm - irqchip : only commit route when irqchip is used
applehttp : Change the variable for stream position in seconds into int64_t
UDF : Close small mem leak in udf_find_entry ()
x86 / amd - iommu : Don ' t detach device from pt domain on driver unbind
UDF : Close small mem leak in udf_find_entry ()
drm / nv40 / mpeg : fix context handling
Input : ads7846 - remove redundant regulator_disable call
pcmdec : check that channels is valid .
avcodec / pcm : Fix overflow in bitrate computation
movenc : Free the buffer returned by url_close_dyn_buffer , regardless of the size
tta : fix off be 1 error in the end detection .
w32threads : with only 1 thread there cant be a active threading mode .
audemux : Add a sanity check for the number of channels
truespeech : check to make sure channels == 1
tta : check for extradata allocation failure in tta demuxer
kvm - irqchip : only commit route when irqchip is used
avcodec / mjpegdec : Check escape sequence validity
hw / intc / arm_gicv3_common : Give no - migration - shift - bug subsection a needed function
fix memory leak in ReadYCBCRImage as SetImageExtent failure
wmaenc : require a large enough output buffer to prevent overwrites
flvdec : fix creation of lots of phantom data streams
H264 : hotfix for speedloss on frame threading and h264 files with slices .
twinvq : validate that channels is not <= 0
kvm - irqchip : only commit route when irqchip is used
rv10 : consider B frames in low delay streams invalid .
cw1200 : Fix cw1200_debug_link_id
mpc8 : check stream count before accessing stream 1 .
avformat / utils : fix memory leak in avformat_free_context
avformat / asfdec_f : Fix memleak
systemctl : never check inhibitors if - H or - M are used (# 3781 )
hls : fix leaking avio_opts on hls_read_header error
Low : tools : crm_mon - resolve coverity UNINIT error
TG3 : limit reaches - 1
resolved : bugfix of null pointer p -> question dereferencing (# 6020 )
branches / zip : row_sel_sec_rec_is_for_clust_rec (): Augment the bug fix in r918 :
blkcg : don ' t call into policy draining if root_blkg is already gone
mds : handle MMonMap
branches / zip : row_sel_sec_rec_is_for_clust_rec (): Augment the bug fix in r918 :
powerpc / pseries : Fix possible leaked device node reference
tcg - s390 : Icache flush is a no - op .
usb : fusbh200 - hcd : fix error handling in fusbh200_hcd_fusbh200_probe ()
fix leak in cc_mslsa . c
[ trunk ] Fix Heap - based buffer - overflow when decoding openjpeg image
mgr : fix OSDMap notifications
Spit out definition of YY_CHAR early
rbd : Fix leaks in rbd_start_aio () error path
kvm - irqchip : only commit route when irqchip is used
ude_rules . c : fix possible NULL pointer dereference in get_key ()
sd - event : fix theoretical leak on error in event_make_signal_data
x86 : e820 memmap - add checking for NULL early param
Made it so cores or threads can be allocated on
sd - event : fix theoretical leak on error in event_make_signal_data
bus : fix issue with reference counting
[ PATCH ] m32r : __cmpxchg_u32 fix
rbd : Fix leaks in rbd_start_aio () error path
mpc7 : check for allocation failure
af_amix : avoid spurious EAGAIN .
xhci : check for failed dma pool allocation
[ PATCH ] dm table split_args : handle no input
fsdev : fix virtfs - proxy - helper cwd
Made it so cores or threads can be allocated on
mds : handle MMonMap
core : free polyline and polygon_sc points
sysctl : parse all keys in a config file
hls : fix leaking avio_opts on hls_read_header error
Bug 595605 - memory leaks in plug - ins / common / curve - bend . c
kvm - irqchip : only commit route when irqchip is used
repository_head_unborn : clear error when HEAD is unborn
mon : Monitor : get required features from monmap
Bug 595605 - memory leaks in plug - ins / common / curve - bend . c
af_amix : prevent memory leak on error path
Protected against garbage in the CapRTL encoder
drm / nouveau / core : swap the order of imem / fb
jvdec : Return EOF on end of file
jvdec : Return EOF on end of file
mpc8 : check stream count before accessing stream 1 .
rgw / rgw_rest . cc : fix 4K memory leak
indeo5 : check against scaleable frames in non scaleable streams .
ffprobe : fix leak in show_packets ()
opus_pvq : remove unneeded assert
avformat / mux : Apply auto bsfs in av_write_frame () too
ASoC : omap - pcm : off - by - one in ' omap_pcm_limit_supported_formats '
sd - event : fix theoretical leak on error in event_make_signal_data
avcodec / rl2 : clear freed pointers
[ Bluetooth ] Revert session reference counting fix
fstab - generator : fix trivial leak
kvm - irqchip : only commit route when irqchip is used
dv : Initialize encoder tables during encoder init .
pmpdec : make i unsigned , avoid undefined behavior of i ++
g726 : Do not try to decode more than one channel .
dca : Fix uninitialized variable warnings .
Input : amijoy - add missing platform check
ext4 : use stream - alloc when mb_group_prealloc set to zero
dsicinav : Check for out of bounds writes
vmxnet3 : validate interrupt indices read on migration
tta : fix off be 1 error in the end detection .
pinctrl : handle zero found dt pinconfig properties better
h264 : Fix parameters to ff_er_add_slice () call
avformat / adxdec : dont return random data on EOF
af_amix : avoid spurious EAGAIN .
avcodec / jpeglsdec : Check for end of bitstream in ls_decode_line ()
pcmdec : check that channels is valid .
avcodec / mjpegdec : sanity check bits
Made it so cores or threads can be allocated on
IOMX : do not clear buffer if it ' s allocated by component
avformat / librtmp : Fix memory leak if RTMP_ConnectStream () fails
avformat / lrcdec : Fix memory leak in lrc_read_header ()
audemux : Add a sanity check for the number of channels
mpegts : Make sure we don ' t return uninitialized packets
avcodec / dcaadpcm : fix use of uninitialized variable
hevc : use av_mallocz () for allocating tab_ipm
[ SCSI ] fcoe : fix missing error check in call to fcoe_if_init
avformat / ipmovie : Check that the OPCODE_INIT_AUDIO_BUFFERS size is large enough
avformat / msf : Also check the codec tag in probing
ui : Make qemu_default_pixman_format () return 0 on unsupported formats
avcodec / alsdec : Check for overread
ffplay : zero initialize copy avpacket
libavcodec / libtwolame : fix null pointer dereference
path - lookup : use correct cleanup function for strv (# 5007 )
H264 : hotfix for speedloss on frame threading and h264 files with slices .
avcodec / utils : Check close before calling it
x86 : e820 memmap - add checking for NULL early param
examples / muxing : check on frame
Made it so cores or threads can be allocated on
lavf / utils : ignore subtitle timestamps in timings calculation
mwifiex : using right aid value for tdls action frame
vmxnet3 : validate interrupt indices read on migration
vaapi_encode : Discard output buffer if picture submission fails
avformat / vqf : Use 64bit for ret to avoid overflow
ipv6 - multicast : Fix memory leak in input path .
avformat / lrcdec : Fix memory leak in lrc_read_header ()
txddemux : Limit allocated packets to filesize .
sapenc : Free AVStream -> info on cleanup
pngdec : do not release buffer on failure instead report full progress
avformat / utils : Check AVFormatContext -> duration in compute_chapters_end () before use
ape : Use the proper variable type
avcodec / snowdec : Check qbias
avcodec / dvdsub_parser : never return 0 when the input isnt 0
wmapro : error out on impossible scale factor offsets
aacdec : free frame buffer if no audio was decoded
Fix out of bound reads in the QDM2 decoder .
vmxnet3 : validate interrupt indices read on migration
Made it so cores or threads can be allocated on
flacdec : silence several " warning : X may be used uninitialized in this function "
avcodec / mpeg4videodec : Check for bitstream overread in decode_vol_header ()
wmaenc : require a large enough output buffer to prevent overwrites
ALSA : lx646es : Fix possible uninitialized variable reference
vmxnet3 : validate interrupt indices read on migration
idcin : check chunk_size value before using it
avcodec / utils : Fix signed integer overflow in rc_initial_buffer_occupancy initialization
matroskadec : unset matroska -> done when seeking
cdxl demux : do not create packets with uninitialized data at EOF .
dca : Fix uninitialized variable warnings .
checkasm : v210 : Fix array overwrite
qemu - io : fix a memory leak
peephole optimization for LOADNIL before STRCAT ; ref # 3110
ffprobe : fix NULL pointer dereference in writer_close ()
fix double free , priv_data is freed in av_open_input_stream
avcodec / mpeg4videoenc : Use 64 bit for times in mpeg4_encode_gop_header ()
indeo4 : fix out - of - bounds function call .
avcodec / pngdec : Clean up on av_frame_ref () failure
h264 : check that execute_decode_slices () is not called too many times
ff_gen_search : make step 64it to prevent hypothetical integer overflow
mp3dec : Fix possibly exploitable crash
mxf : Fix a possible leak of extradata
libopusenc : check return value
flvenc : silence bogus warning
wmapro : error out on impossible scale factor offsets
avcodec / lagarith : disable lag_decode_zero_run_line () and ask for a sample
bus : properly unset default bus pointer when destroying last reference
ffprobe : replace av_destruct_packet () with av_free_packet ()
tmpfiles : restore previous behavior for F / f
avcodec / mpeg4videodec : Check for bitstream overread in decode_vol_header ()
avcodec / jpeg2000 : Fixes integer overflow in ff_jpeg2000_ceildivpow2 ()
svq3 : propagate codec memory allocation failure in context init
avformat / pcmdec : fix memory leak
indeo4 : update AVCodecContext width / height on size change
cherokee_buffer_crc32 () did not handle empty buffers correctly .
dsicin : Add some basic sanity checks for fields read from the file
avprobe : free options at the end of main ()
avformat / 4xm : shrink packet if it was only partially initialized
vc1dec : prevent memory leak in error path
avcodec / wmalosslessdec : fix mclms_coeffs * array size
avcodec / wavpack : Fix runtime error : left shift of negative value - 1
avformat / oggparseogm : check input size before reading t
avformat / utils : fix memory leak in avformat_free_context
dca : Fix uninitialized variable warnings .
pcmdec : check that channels is valid .
msg : initialize footer
ARM : zynq : initialize slcr mapping earlier
rtsp : Free the rtpdec context properly
avcodec / lagarith : disable lag_decode_zero_run_line () and ask for a sample
mpeg12enc : Don ' t set up run - level info for level 0 .
avcodec / mjpegdec : sanity check bits
Bugfix : Free shares discarded before beginning submission
lavfi / hqdn3d : avoid use of uninitialized variable .
ui : Make qemu_default_pixman_format () return 0 on unsupported formats
avformat / dfa : check first frame offset in probe
usbutils - which_intinfo not requried
png : Use av_freep () instead of av_free ()
avformat / lrcdec : Fix memory leak in lrc_read_header ()
vmxnet3 : validate interrupt indices read on migration
fix mem leak
asfenc : fix leaking asf -> index_ptr on error
flvdec : fix creation of lots of phantom data streams
ffv1 : set internal frame defaults
auth : oauth2 - introspection lookup error should be internal failure
range check - set_default_value
avformat / movenc : Write version 2 of audio atom if channels is not known
ttadec : check channel count as read from extradata .
wmadec : Verify bitstream size makes sense before calling init_get_bits .
id3v2 : catch avio_read errors in check_tag
mpegvideo : Avoid 32 - bit wrapping of linesize multiplications
arm64 : Move BP hardening to check_and_switch_context
Initialize a variable in all cases
avformat / vqf : check avio_read () s return value more completely
Input : amijoy - add missing platform check
rtsp : Free the rtpdec context properly
adxenc : check return value of avcodec_alloc_frame ()
qemu - io : fix a memory leak
hwcontext_dxva2 : initialize D3DDISPLAYMODEEX correctly
flvdec : fix creation of lots of phantom data streams
avcodec / rl2 : clear freed pointers
* examples / fadot . c : free the result from FA_AS_REGEXP
avcodec / mlpdec : Do not leave a invalid num_primitive_matrices in the context
dca : Fix uninitialized variable warnings .
avcodec / mpegaudiodec_template : Make l3_unscale () work with e = 0
drm / i915 : check for supported depth at fb init time
mxfdec : Fix files > 2 GiB
drm / i915 : Only flush fbc on sw when fbc is enabled .
iio : Accept a leading '+' sign when parsing fixed point numbers
motionpixels : check extradata size
targa : prevent integer overflow in bufsize check .
matroskadec : unset matroska -> done when seeking
fix mem leak
sysctl : require CAP_SYS_RAWIO to set mmap_min_addr
dca : Fix uninitialized variable warnings .
avcodec / hnm4video : check that packetsize is at least 8
[ Bluetooth ] Revert session reference counting fix
h264 : Fix a possible overread in decode_nal_units ()
fix heap - buffer - overflow in mp4read . c
cbs_h2645 : Fix memory leak on when reading SEI fails
tta : check for extradata allocation failure in tta demuxer
pcmdec : check that channels is valid .
dsicin : Add some basic sanity checks for fields read from the file
fraps : check for overread
fastmemcpy path fixed
Bugfix : Free shares discarded before beginning submission
[ PATCH ] libata : fast exit from EH while unloading
avcodec / ffv1dec : Check for 0 quant tables
avformat / apngdec : Use 64bit for ret to avoid overflow
lavf / pcm : check size , do not produce invalid packets
fix mem leak
libavformat : Check for malloc failures in avformat_new_stream
blkid : fix typo in ID_FS_AMBIVALEN =
staging : vt6656 : [ BUG ] out of bound array reference in RFbSetPower .
avcodec / smc : fix off by 1 error
lavfi / alphamerge : fix leak introduced in 217163eb
indeo5 : Fix null pointer dereference
avcodec / jpeg2000 : Fixes integer overflow in ff_jpeg2000_ceildivpow2 ()
Use 16pix icons for size < 19
avcodec / hnm4video : check that packetsize is at least 8
Fixed another OOM problem , this time with test 64 .
ppc : fix crash in ppc system single step support
mov : reset dref_count on realloc to keep values consistent .
branches / zip : row_sel_sec_rec_is_for_clust_rec (): Augment the bug fix in r918 :
Input : amijoy - add missing platform check
amd - xgbe : Check for complete packet on skb allocation error
avcodec / mpeg4videodec : Check for bitstream overread in decode_vol_header ()
asrc_aevalsrc : Fix use of uninitialized pointer inside av_strtok ()
Check for out of bound writes when building tree in bink decoder .
clk : rockchip : rename rga clock - id on rk3399
vmxnet3 : Do not fill stats if device is inactive
qemu - sockets : Plug fd leak on unix_connect_opts () error path
block : Limit size to INT_MAX in bdrv_check_byte_request ()
vhost / scsi : call vhost_dev_cleanup () at unrealize () time
kvm - irqchip : only commit route when irqchip is used
acpi : add dummy write function for acpi timer
qcow2 : Use BB for resizing in qcow2_amend_options ()
qdev - monitor : Fix crash when device_add is called with abstract driver
iio : Accept a leading '+' sign when parsing fixed point numbers
Bug 595605 - memory leaks in plug - ins / common / curve - bend . c
tap : clear vhost_net backend on cleanup
Made it so cores or threads can be allocated on
avprobe : free options at the end of main ()
OMAPDSS : HDMI : init output earlier
branches / zip : row_sel_sec_rec_is_for_clust_rec (): Augment the bug fix in r918 :
block : Limit size to INT_MAX in bdrv_check_byte_request ()
hw / intc / arm_gicv3_common : Give no - migration - shift - bug subsection a needed function
kvm - irqchip : only commit route when irqchip is used
* examples / fadot . c : free the result from FA_AS_REGEXP
modules / usrloc : init _ul_sruid also in mi_child_init
Made it so cores or threads can be allocated on
cadence_uart : bounds check write offset
modules / usrloc : init _ul_sruid also in mi_child_init
ide : Remove unneeded null pointer check
kvm - irqchip : only commit route when irqchip is used
win32 - aio : Fix memory leak
drivers / net / phy / mdio_bus . c : call put_device on device_register () failure
[ media ] cec : ignore messages when log_addr_mask == 0
Fixed potential buffer overflow in ` ProcessRadioRxDone `
avcodec / lagarith : Fix runtime error : left shift of negative value - 1
vmxnet3 : validate queues configuration read on migration
hw / nvram / spapr_nvram : Device can not be created by the users
tools : iio : generic_buffer : free dev_dir_name on exit
Input : wistron_btns - fix a memory leak in wb_module_init error path
mem : Consistently return NULL for av_malloc ( 0 )
hw / arm / digic : Mark device with user_creatable = false
branches / zip : row_sel_sec_rec_is_for_clust_rec (): Augment the bug fix in r918 :
peephole optimization for LOADNIL before STRCAT ; ref # 3110
tcm_vhost : Initialize vq -> last_used_idx when set endpoint
usb : sanity check setup_index + setup_len in post_load
vmxnet3 : validate interrupt indices read on migration
rbd : Fix leaks in rbd_start_aio () error path
qdev : Fix crash by validating the object type
[ media ] em28xx : fix em28xx - input removal
vmxnet3 : validate interrupt indices read on migration
Modified various APIs to match our intent .
[ media ] em28xx : fix em28xx - input removal
target - s390x : Mark check_privileged () as ! CONFIG_USER_ONLY
qcow2_check : Mark image consistent
usb : early : fixed coding style issue related to : operator
Made it so cores or threads can be allocated on
Made it so cores or threads can be allocated on
ACPI : thinkpad - acpi : do not arm fan watchdog if it would not work
tpm_tis : Return TPM_VERSION_UNSPEC in case of BE failure
usb - redir : Fix crash on migration with no client connected
qemu : msix nit : clear msix_entries_nr on error
ARM : zynq : initialize slcr mapping earlier
vmscan : promote shared file mapped pages
blkverify : Fix BDS leak in . bdrv_open error path
xen_disk : mark ioreq as mapped before unmapping in error case
ACPICA : Fix for Linux source translation
[ media ] rtl28xxu : lower the rc poll time to mitigate i2c transfer errors
ibm_newemac : Fixes entry of short packets
qemu - img convert : Use cache = unsafe for output image
hw / arm : Mark the " fsl , imx31 " device with user_creatable = false
branches / zip : row_sel_sec_rec_is_for_clust_rec (): Augment the bug fix in r918 :
hls : fix leaking avio_opts on hls_read_header error
mpegts : Make sure we don ' t return uninitialized packets
qdev : Fix crash by validating the object type
powerpc / powernv : Initialize DMA for PEs
rtl8139 : skip offload on short TCP header ( CVE - 2015 - 5165 )
target / ppc : fix cpu_ov setting for 32 - bit
support sucking data from streams in file_put_contents ()
posix - aio - compat : Plug memory leak on paio_init () error path
kvm - irqchip : only commit route when irqchip is used
Fix a read past end of buffer crash in the mp3 probe
hw / intc / arm_gicv3_common : Give no - migration - shift - bug subsection a needed function
[ SCSI ] megaraid : Use the proper type to hold the irq number .
Made it so cores or threads can be allocated on
Drivers : hv : balloon : Make the balloon driver not unloadable
iio : gyro : ADXRS450 : Add missing read support for calibbias attribute
hw / arm / digic : Mark device with user_creatable = false
cw1200 : Fix cw1200_debug_link_id
pci : deassert intx when pci device unrealize
x86 / smpboot : Move data structure to its primary usage scope
KVM : PIT : control word is write - only
ppc : fix monitor access to CR
vt82c686 : avoid out - of - bounds read
applehttp : Avoid reading uninitialized memory
powerpc / crypto / crc32c - vpmsum : Fix missing preempt_disable ()
kvm - irqchip : only commit route when irqchip is used
Protected against garbage in the CapRTL encoder
qemu - sockets : use qapi_free_SocketAddress in cleanup
qom : fix leak for objects created with - object
Made it so cores or threads can be allocated on
dmaengine : pl330 : Check if the DMA descriptor is NULL
staging : comedi : adv_pci_dio : do board reset early in (* auto_attach )
IIO : Documentation : generic_buffer example : Avoid NULL pointer dereference
KVM : PIT : control word is write - only
ACPI : thinkpad - acpi : do not arm fan watchdog if it would not work
qcow2 : Free allocated snapshot table on error
parallels : wrong call to bdrv_truncate
qcow2 : Free allocated snapshot table on error
hostmem - file : plug a small leak
avformat / bintext : protect against potential overflow of chars_per_frame
branches / zip : row_sel_sec_rec_is_for_clust_rec (): Augment the bug fix in r918 :
Modified various APIs to match our intent .
resolved : merge two bools into a bitfield
hls : fix leaking avio_opts on hls_read_header error
hostmem - file : plug a small leak
counters : don ' t run if no counters have been registered
systemctl : plug a leak upon create_symlink mismatch
x86 / boot : Fix minor fd leakage in tools / relocs . c
lib : Add asserts to t_abspath and t_abspath_to
libudev : fix sysnum logic for digit - only device names
avcodec / mpegaudiodec_template : Make l3_unscale () work with e = 0
base - filesystem : avoid all searching if the link already exists
cgroup : the root cgroup is always populated
Input : amijoy - add missing platform check
journalctl : show fields requested with -- field in full
Made it so cores or threads can be allocated on
added missing g_free ()
avformat / mov : fix integer overflow
hls : fix leaking avio_opts on hls_read_header error
locale : make sure that l is freed
[ sgen / mono ] Don ' t leak stack info struct
sd - event : fix theoretical leak on error in event_make_signal_data
util : make sure we don ' t clobber errno in error path
Modified various APIs to match our intent .
Remove the temporary file created by mysql_upgrade
avcodec / g2meet : Also clear tile dimensions on header_fail
branches / zip : Check for autoinc next value overflow explicitly when offset
dns ( tcp ) - fix coverity cid 1374307
Applying InnoDB snapshot
[ PATCH ] um : fix - up schedule_timeout () usage
make macfb_setup () static
basic : use for () loop instead of while ()
mds : fix can_rdlock_soon vs xlock on FileLock
mk_core : thread : fix ucontext . h path
[ S390 ] idle : avoid RCU usage in extended quiescent state
jvdec : Return EOF on end of file
kvm - irqchip : only commit route when irqchip is used
Set the WSQ TLS key to inited after initing it .
Made it so cores or threads can be allocated on
Protected against garbage in the CapRTL encoder
Spit out definition of YY_CHAR early
Fix the | name | of PerformanceNavigationTiming
hls : fix leaking avio_opts on hls_read_header error
hw / intc / arm_gicv3_common : Give no - migration - shift - bug subsection a needed function
opm : properly disable opm on blacklist deletions .
kvm - irqchip : only commit route when irqchip is used
hwmon : ( emc1403 ) Support full range of known chip revision numbers
ARM : zynq : initialize slcr mapping earlier
rbd : Fix leaks in rbd_start_aio () error path
exit - on - idle : only exit if actually idle
drm / ttm : fix incorrect logic in ttm_bo_io path
support sucking data from streams in file_put_contents ()
Input : amijoy - add missing platform check
[ scheduler ] Remove implicit fallthrough in switch
Modified various APIs to match our intent .
Made it so cores or threads can be allocated on
Fixed potential buffer overflow in ` ProcessRadioRxDone `
tap : clear vhost_net backend on cleanup
avcodec / aacdec_fixed : fix : left shift of negative value - 1
kvm - irqchip : only commit route when irqchip is used
arm64 : drop unnecessary cache + tlb maintenance
kvm - irqchip : only commit route when irqchip is used
[ media ] ascot2e : Fix I2C message size check
avutil / tree : fix memleaks
Bug 595605 - memory leaks in plug - ins / common / curve - bend . c
ASoC : rockchip : Enable 192khz in hw_params operation
jvdec : Return EOF on end of file
staging : comedi : ni_at_ao : fix namespace clash of ' RSI ' define
drm / i915 : Only clear the GPU domains upon a successful finish
cdc_ncm : return - ENOMEM if kzalloc fails
rgw : Error check on return of read_line ()
* examples / fadot . c : free the result from FA_AS_REGEXP
iio : Accept a leading '+' sign when parsing fixed point numbers
avformat / matroskadec : fix integer overflow
usb : isp1760 : check for null return from kzalloc
trace : break circular dependency in event - internal . h
lib - smtp : smtp_server_connection_pending_command_data () - don ' t crash if smtp_parser is NULL
mwifiex : block data traffic to tx paused receive address
Fixed debug . c to work again unchanged .
apetag : propagate errors .
qsvdec : Fix memory leak
resolved : bugfix of null pointer p -> question dereferencing (# 6020 )
fetch - pack : Finish negotation if remote replies " ACK % s ready "
fw_cfg : add required header files
io : assert errors before asserting content in I / O test
Protected against garbage in the CapRTL encoder
branches / zip : row_sel_sec_rec_is_for_clust_rec (): Augment the bug fix in r918 :
ACPICA : Fix for Linux source translation
Made it so cores or threads can be allocated on
drm / i915 / execlists : Add a GEM_TRACE to show when the context is completed
journal - upload : allow the tool to start
src / pcm . c : Improve error reporting .
hw / pc_sysfw : Fix memory leak
seq_file : disallow extremely large seq buffer allocations
vmxnet3 : validate queues configuration read on migration
check that ctx is not a null ptr before deref cid # 140
Check if auth_user is set .
avformat / dss : Use av_freep () to avoid leaving stale pointers in memory
vmxnet3 : validate interrupt indices read on migration
Made it so cores or threads can be allocated on
libvncclient : free vncRec memory in rfbClientCleanup ()
Initialize a variable in all cases
dns ( tcp ) - fix coverity cid 1374307
perf_events : Fix invalid pointer when pid is invalid
riscv : Return actual instruction length
UDF : Close small mem leak in udf_find_entry ()
kvm - irqchip : only commit route when irqchip is used
kvm - irqchip : only commit route when irqchip is used
Applying InnoDB snapshot
TG3 : limit reaches - 1
x86 / mce / AMD : Make the init code more robust
extcon : Fix return value in extcon - class . c
Copy the silence flags when sending update notifications .
ACPICA : Fix for Linux source translation
doc / examples : Handle new types in avio_list_dir
Added assistance file name NULL check .
spi : add spi_message_init_no_memset to avoid zeroing the spi_message
librados : fix use without NULL check in rados_pool_list
ceph : add missing locking to protect i_snap_realm_item during split
KVM : PIT : control word is write - only
Bugfix : Free shares discarded before beginning submission
systemctl : fix memory leak in systemctl_parse_argv
staging : comedi : ni_at_ao : fix namespace clash of ' RSI ' define
Input : amijoy - add missing platform check
um : Missing pipe handling
basic : use for () loop instead of while ()
avformat / 4xm : shrink packet if it was only partially initialized
ACPICA : Fix for Linux source translation
Fixed a bug in vis . c
followup to SET PASSWORD fix
PM : Free memory bitmaps if opening / dev / snapshot fails
avformat / mpegtsenc : Changed Video PES packet length to 0 .
Input : ati_remote - add missing input_sync ()
mtd : mtd - abi : add a helper to detect the nand type
drm / radeon : don ' t setup audio on asics that don ' t support it
Made it so cores or threads can be allocated on
iommu / exynos : Add / remove callbacks should fail if no iommu is available
kvm : irqchip : fix memory leak
regmap : cache : Don ' t attempt to sync non - writeable registers
igmp : Avoid zero delay when receiving odd mixture of IGMP queries
bridge : cleanup : remove unneed check
ARM : S3C64XX : Remove unused CPU devices for Cragganmore
HID : hidraw , fix a NULL pointer dereference in hidraw_ioctl
[ SCSI ] qla2xxx : Process DPC requests within valid Fabric topologies .
Made it so cores or threads can be allocated on
[ Bluetooth ] Revert session reference counting fix
rtlwifi : rtl8192ce : Set fw_ready flag
vmxnet3 : validate interrupt indices read on migration
ARM : zynq : initialize slcr mapping earlier
x86 : e820 memmap - add checking for NULL early param
[ media ] davinci : vpbe : fix build error when CONFIG_VIDEO_ADV_DEBUG is enabled
PCI : xilinx : Dispose of MSI virtual IRQ
Fixed bug in sub select after last merge
kvm - irqchip : only commit route when irqchip is used
jbd2 : clear BH_Delay & BH_Unwritten in journal_unmap_buffer
hwmon : ( adm1025 ) Fix vrm write operation
[ media ] coda : v4l2 - compliance fix : zero pixel format priv field
rtc : ac100 : Add NULL checking for devm_kzalloc call
KVM : Grab pic lock in kvm_pic_clear_isr_ack
avcodec / hevc : make * ps_id unsigned
[ SCSI ] ibmvscsi : Correct DMA mapping leak
fb_defio : Push file_update_time () into fb_deferred_io_mkwrite ()
[ SCSI ] be2iscsi : memset wrb for ring create
ath6kl : Support net_stats . multicast
amd - xgbe : Check for complete packet on skb allocation error
ALSA : es1688 : Use strcpy () instead of sprintf ()
kvm - irqchip : only commit route when irqchip is used
drivers / net / eql . c : prevent reading uninitialized stack memory
kvm : x86 : fix emulator buffer overflow ( CVE - 2014 - 0049 )
appletalk : fix atalk_getname () leak
Input : amijoy - add missing platform check
drm / tilcdc : Remove unnecessary braces
kvm - irqchip : only commit route when irqchip is used
drm / nv40 / mpeg : fix context handling
stdio_pclose : return - errno on error ( v3 )
[ media ] coda : v4l2 - compliance fix : zero pixel format priv field
iio : chemical : ccs811 : Corrected firmware boot / application mode transition
Set the WSQ TLS key to inited after initing it .
ath9k : Fix a crash in 2 WIRE btcoex chipsets
hub : debug message for failing to enable device
Modified various APIs to match our intent .
staging : media : lirc : Use pr_ * instead of printk
staging : comedi : cb_pcidas : check for failure of subdev_8255_init
i2c : i2c - bcm - kona : Fix module build
e1000e : magic number cleanup - ETH_ALEN
ipv6 : udp : fix the wrong headroom check
[ Bluetooth ] Revert session reference counting fix
vmxnet3 : validate interrupt indices read on migration
staging : dgap : Add check for MAXBOARDS in . probe ( dgap_init_one )
drm / fb - helper : Do the ' max_conn_count ' zero check
Btrfs : Fix balance panic
mwifiex : pcie : Fix memory leak in mwifiex_pcie_alloc_cmdrsp_buf
ipv4 : Use FIELD_SIZEOF () in inet_init ().
imx - drm : ipu - dp : Check the return value of devm_kzalloc ()
Bugfix : Free shares discarded before beginning submission
ath9k_htc : Protect ampdu_action with a mutex
vmxnet3 : validate interrupt indices read on migration
[ ARM ] pxa : add missing irq events for pxa168
atl1c : return PCI_ERS_RESULT_DISCONNECT on permanent failure
KVM : Improve create VCPU parameter ( CVE - 2013 - 4587 )
Staging : Fix leak in drivers / staging / at76_usb . c
[ Bluetooth ] Revert session reference counting fix
drm / radeon : unlock the ring mutex while waiting for the next fence
fuse : reset waiting
usb : core : check for valid id_table when using the RefId feature
HID : hyperv : match wait_for_completion_timeout return type
peephole optimization for LOADNIL before STRCAT ; ref # 3110
kvm : irqchip : fix memory leak
perf session : Skip event correctly for unknown id / machine
i2c : omap : on -> remove () call pm_runtime_put_sync ()
[ ALSA ] soc - 0 . 13 ASoC DAPM bug fix for unnamed streams
xfrm : Restrict extended sequence numbers to esp
[ JFFS2 ] Add declaration of jffs2_lzo_ { init , exit } to compr . h
irqchip / gic - v3 : Fix out - of - bound access in gic_set_affinity
video : wm8505fb : Remove redundant of_match_ptr
ALSA : bebob : Fix a missing to unlock mutex in error handling case
KVM : nSVM : Fix IOIO size reported on emulation
UDF : Close small mem leak in udf_find_entry ()
[ media ] coda : add compressed flag to format enumeration output
ath9k : Check virtual wiphy state on tx ()
ptr_ring : fail early if queue occupies more than KMALLOC_MAX_SIZE
Input : gpio_keys - fix a memory leak
vmxnet3 : validate interrupt indices read on migration
hls : fix leaking avio_opts on hls_read_header error
ipv6 : Export nf_ct_frag6_gather ()
sched : Try to deal with low capacity , fix update_sd_power_savings_stats ()
hwmon : ( f75375s ) Disable setting DC fan control mode for F75373
kvm - irqchip : only commit route when irqchip is used
fix BSONObj ownership
vmxnet3 : validate interrupt indices read on migration
[ media ] s5p - fimc : Fix return value of __fimc_md_create_flite_source_links ()
f2fs : persist system . advise into on - disk inode
staging : dwc2 : fix thinko in dwc2_hc_set_even_odd_frame ()
arm64 : Move BP hardening to check_and_switch_context
[ Bluetooth ] Revert session reference counting fix
perf tools : Try to lookup kernel module map before creating one
rtl8150 : rtl8150_disconnect (...) does not need tasklet_disable (...)
drm / amdgpu_gem : fix error handling path in amdgpu_gem_va_update_vm
libata : drop WARN from protocol error in ata_sff_qc_issue ()
avr32 : Fix sigaltstack behaviour
[ media ] v4l : vb2 - dma - contig : align buffer size to PAGE_SIZE
sysusers : fix memleak (# 4430 )
iwlwifi : mvm : Avoid signal based decisions if ave beacon RSSI is 0
md : raid1 : kill warning on powerpc_pseries
rpmsg : virtio : Fix possible double free in rpmsg_virtio_add_ctrl_dev ()
drm / i915 : Only flush fbc on sw when fbc is enabled .
vmxnet3 : validate interrupt indices read on migration
hls : fix leaking avio_opts on hls_read_header error
exit - on - idle : only exit if actually idle
[ Bluetooth ] Revert session reference counting fix
kvm - irqchip : only commit route when irqchip is used
perf bench : Fix NULL pointer dereference in " perf bench all "
[ PATCH ] pcnet32 : remove incorrect pcnet32_free_ring
btrfs : reada : move reada_extent_put to place after __readahead_hook ()
libtcmu : free ctx in tcmulib_initialize
hostmem - file : plug a small leak
UDF : Close small mem leak in udf_find_entry ()
ALSA : sb16 - info leak in snd_sb_csp_ioctl ()
proc : Point / proc / net at / proc / thread - self / net instead of / proc / self / net
[ media ] coda : v4l2 - compliance fix : zero pixel format priv field
firewire : core : remove an unnecessary zero initialization
drm / i915 : take struct mutex around fb unref
Hardware : Fix possible recursive locking in hwregs . c
PCI : hv : Use the correct buffer size in new_pcichild_device ()
skge : Unmask interrupts in case of spurious interrupts
rpmsg : virtio : Fix possible double free in rpmsg_probe ()
rbd : Fix leaks in rbd_start_aio () error path
gpio : adp5588 : remove unnecessary OOM messages
staging : comedi : cb_das16_cs : return ' 0 ' for successful attach
Bug 595605 - memory leaks in plug - ins / common / curve - bend . c
Input : twl4030 - pwrbutton - fix a leak of the IRQ during init failure
acpi : add checking for NULL early param
x86 : add dummy acpi_numa_processor_affinity_init () implementation on 32 - bit
ASoC : WM8903 : Free IRQ on device removal
dm raid1 : switch rh_in_sync to blocking in do_reads
vmxnet3 : validate interrupt indices read on migration
[ media ] lirc : return ENOTTY when device does support ioctl
vf_hqdn3d : make config_props work properly when called multiple times .
[ media ] xc5000 : use after free in release ()
PCI : hv : Use the correct buffer size in new_pcichild_device ()
drm / nouveau / devinit / nv04 : change owner to int
kvm - irqchip : only commit route when irqchip is used
UBIFS : replace count * size kzalloc by kcalloc
drm / i915 / cnl : Reprogram DMC firmware after S3 / S4 resume
diff : avoid potential null pointer dereference on error
sd - event : fix theoretical leak on error in event_make_signal_data
KVM : X86 : Don ' t use PV TLB flush with dedicated physical CPUs
x86 / smpboot : Move data structure to its primary usage scope
branches / zip : row_sel_sec_rec_is_for_clust_rec (): Augment the bug fix in r918 :
iio : core : Handle error when mask type is not separate
drm / helper : lock all around force mode restore
i40e : Add missing braces to i40e_dcb_need_reconfig ()
arch / powerpc / sysdev / ehv_pic . c : add missing kfree
kvm - irqchip : only commit route when irqchip is used
HID : hyperv : match wait_for_completion_timeout return type
[ PATCH ] md : Avoid oops when attempting to fix read errors on raid10
vfio - pci : Add missing break to enable VFIO_PCI_ERR_IRQ_INDEX
ov7670 : clean up ov7670_read semantics
ASoC : img - parallel - out : Add pm_runtime_get / put to set_fmt callback
iomap : prepare iomap_fiemap for attribute mappings
check that ctx is not a null ptr before deref cid # 140
[ SCSI ] ibmvfc : Fix DMA mapping leak on memory allocation failure
modules / usrloc : init _ul_sruid also in mi_child_init
[ media ] vb2 : fix compiler warning
imx - drm : ipu - dp : Check the return value of devm_kzalloc ()
ALSA : hda - Update automute / automic upon jack retasking
dxva : DXVA2_ModeHEVC_VLD_Main10 does not support Main
sched : reorder struct sched_rt_entity to remove padding on 64 bit builds
md : check for memory allocation failure in faulty personality
ASoc : wm8731 : add 32bit mode .
carl9170 : fix typo in PS code
usb : dwc3 : gadget : add a debugging print when initializing endpoints
Added missing DBUG_RETURN
mss1 : improve check if decoded pivot is invalid
gro : Reset dev pointer on reuse
x86 / smpboot : Move data structure to its primary usage scope
Protected against garbage in the CapRTL encoder
TG3 : limit reaches - 1
mtd : onenand : omap2 : Remove skip initial unlocking support
fstab - generator : fix trivial leak
[ PATCH ] Bluetooth : fix potential NULL ptr deref in dtl1_cs . c :: dtl1_hci_send_frame ()
mtd : onenand : omap2 : Remove skip initial unlocking support
mtd : onenand : omap2 : Remove skip initial unlocking support
tracepoints : use rcu_ * _sched_notrace
mfd : palmas : Initialise client -> of_node for dummy created client
ARM : EXYNOS : Fix potential NULL pointer access in exynos_sys_powerdown_conf
fs / binfmt_misc . c : add terminating newline to / proc / sys / fs / binfmt_misc / status
drm / radeon : fix UVD 256MB check
drm / i915 : Make disable_cursor_plane similar to commit_cursor_plane .
ff_h264_direct_ref_list_init : fix B slice check .
drm / ast : Use lockless gem BO free callback
USB : uas : Fix slave queue_depth not being set
Made it so cores or threads can be allocated on
irqchip / gic - v3 : Fix out - of - bound access in gic_set_affinity
support sucking data from streams in file_put_contents ()
ocfs2 : honor O_ ( D ) SYNC flag in fallocate
ALSA : usb - audio : add DSD support for new Amanero PID
iommu / exynos : Add / remove callbacks should fail if no iommu is available
staging : slicoss : Fix possible reuse of freed memory in timer function
vmxnet3 : validate interrupt indices read on migration
ASoc : wm8731 : add 32bit mode .
kvm - irqchip : only commit route when irqchip is used
drm / i915 : switch to default context on idle
drm / radeon / kms : add pll debugging output
Applying InnoDB snapshot
drm / i915 : Fix intel_tile_width () parameters
Input : gpio_keys - fix a memory leak
mnt : release locks on error path in do_loopback
branches / zip : row_sel_sec_rec_is_for_clust_rec (): Augment the bug fix in r918 :
[ Bluetooth ] Revert session reference counting fix
sd - event : fix theoretical leak on error in event_make_signal_data
ath9k : set MAC version for AR9550
mfd : db8500 - prcmu : Check return of devm_ioremap for error
drm / i915 : Avoid NULL ctx -> obj dereference in debugfs / i915_context_info
drm / nv40 / mpeg : fix context handling
i40e : validate VSI param from VFs
staging : comedi : adv_pci_dio : do board reset early in (* auto_attach )
target - i386 : cpu : Fix potential buffer overrun in get_register_name_32 ()
cxlflash : Fix to prevent stale AFU RRQ
mtd : socrates_nand : Remove redundant dev_set_drvdata
clk : rockchip : rename rga clock - id on rk3399
staging : olpc_dcon : actually return the value of i2c_add_driver
sky2 : Shut off interrupts before NAPI
fix mem leak
[ IPV4 ] fib_trie : fix duplicated route issue
wl12xx : protect wl12xx_op_set_rts_threshold ()
qdev - monitor : Fix crash when device_add is called with abstract driver
h264 : Fix memory leak on ff_h264_decode_init () failure
gpio / ep93xx : Use irq_set_handler_locked ()
drm / amdgpu / gfx9 : bypass clockgating setting
hv_netvsc : fix a race between netvsc_send () and netvsc_init_buf ()
ppc : Let kvmppc_reset_htab () return 0 for ! CONFIG_KVM
ACPICA : debugger : fix memory leak on Pathname
tool_writeout : fixed a buffer read overrun on -- write - out
vmxnet3 : validate interrupt indices read on migration
staging : comedi : adv_pci_dio : do board reset early in (* auto_attach )
irqchip / gic - v3 : Fix out - of - bound access in gic_set_affinity
UIO : uio_pdrv : fix memory leak
Bugfix : Free shares discarded before beginning submission
readd early log level check for performance reasons .
irqchip / gic - v3 : Fix out - of - bound access in gic_set_affinity
imx - drm : imx - drm - core : improve safety of imx_drm_add_crtc ()
drm / fb - helper : Do the ' max_conn_count ' zero check
Input : usbtouchscreen - add support for LG Flatron T1710B
[ SCSI ] zfcp : Fix error checking for ELS ADISC requests
mac80211 : mesh only parameter mppath maybe unused
[ PATCH ] mv643xx_eth : Update dev -> last_rx on packet receive
Input : polldev - immediately poll device upon opening
[ SCSI ] libsas : remove expander from dev list on error
mem : Consistently return NULL for av_malloc ( 0 )
gpio : adp5588 : remove unnecessary OOM messages
ALSA : usb : caiaq : check for cdev -> n_streams > 1
UBIFS : return EROFS in case of broken commit
pinctrl : s3c24xx : off by one in s3c24xx_eint_init ()
add svn_version () and global var which are
iio : lmp91000 : Set parent device
Btrfs : fix em leak in find_first_block_group
[ media ] coda : add compressed flag to format enumeration output
drivers / char / hpet . c : fix information leak to userland
Btrfs : remove dead code
drm : stop information leak of old kernel stack .
[ PATCH ] stack overflow safe kdump : safe_smp_processor_id (): voyager
usbdev : BKL pushdown
rt2x00 : Shortcut link state updates when not operating as STA
dmaengine : k3dma : Off by one in k3_of_dma_simple_xlate ()
Page migration : Do not accept invalid nodes in the target nodeset
drm / i915 / skl : Restore the DDI translation tables when enabling PW1
UBI : always re - read in case of read failures
ASoC : WM8903 : Free IRQ on device removal
drivers / perf : arm_pmu_acpi : Release memory obtained by kasprintf
panic , vt : do not force oops output when panic_timeout < 0
hls : fix leaking avio_opts on hls_read_header error
iio : buffer : Fix NULL pointer deref caused by empty scan mask
x86 / smpboot : Move data structure to its primary usage scope
clk : rockchip : rename rga clock - id on rk3399
mds : handle MMonMap
x86 / smpboot : Move data structure to its primary usage scope
kvm : irqchip : fix memory leak
librados : take lock when signaling notify cond
arm64 : Move BP hardening to check_and_switch_context
kvm - irqchip : only commit route when irqchip is used
[ Bluetooth ] Revert session reference counting fix
dm flakey : check for null arg_name in parse_features ()
jpeg : Fix another possible buffer overrun
KVM : nVMX : Validate the IA32_BNDCFGS on nested VM - entry
[ PATCH ] e1000 : Enable polling before enabling interrupts
vmxnet3 : validate interrupt indices read on migration
bus : properly unset default bus pointer when destroying last reference
mds : fix can_rdlock_soon vs xlock on FileLock
drm / i915 : release mutex in i915_gem_init ()' s error path
i40iw : Add NULL check for ibqp event handler
brstm : fix use of uninitialized variables
Fix out of bound writes in fix_bitshift () of the shorten decoder .
drm / exynos / fbdev : set smem_len for fbdev
vmxnet3 : validate interrupt indices read on migration
[ PATCH ] mv643xx_eth : Update dev -> last_rx on packet receive
x86 / smpboot : Move data structure to its primary usage scope
Input : polldev - immediately poll device upon opening
reiserfs : Don ' t call reiserfs_get_acl () with the reiserfs lock
ipv4 : Limit size of route cache hash table
pinctrl : pinctrl_select_state : set the old_state back on error
[ Bluetooth ] Revert session reference counting fix
ASoC : Make struct snd_soc_card ' s dapm_widgets and dapm_routes const
kvm : irqchip : fix memory leak
ath9k : Do not use local_bh_disable in ampdu_action
[ ALSA ] ca0106 - Add missing sysfs device assignment
drm / amd / display : fix gamma setting
[ Bluetooth ] Revert session reference counting fix
target - i386 : cpu : Fix potential buffer overrun in get_register_name_32 ()
hls : fix leaking avio_opts on hls_read_header error
bnx2x : Add known PHY type check
kvm - irqchip : only commit route when irqchip is used
vmxnet3 : validate queues configuration read on migration
USB : serial : digi_acceleport : fix OOB - event processing
drm / tilcdc : Remove unnecessary braces
drm / i915 / skl : Restore the DDI translation tables when enabling PW1
Hardware : Fix possible recursive locking in hwregs . c
xfrm : Restrict extended sequence numbers to esp
[ SCSI ] scsi_lib : Fix bug in completion of bidi commands
branches / zip : row_sel_sec_rec_is_for_clust_rec (): Augment the bug fix in r918 :
virtio : console : Make write () return - ENODEV on hot - unplug
kvm - irqchip : only commit route when irqchip is used
kvm : irqchip : fix memory leak
spi : qup : Fix return value checking for pm_runtime_get_sync ()
staging : comedi : adv_pci_dio : do board reset early in (* auto_attach )
x86 / smpboot : Move data structure to its primary usage scope
dm cache : fix error return code in cache_create
pty : fix data loss when stopped (^ S /^ Q )
adm8211 : fix cfg80211 band API conversion
[ IPSEC ] Set byid for km_event in xfrm_get_policy
x86 apic_32 . c section fix
[ ALSA ] soc - Report errors from snd_soc_dapm_set_endpoint ()
iommu / exynos : Add / remove callbacks should fail if no iommu is available
tracing : Fix off - by - one in trace_get_user ()
i2c : i2c - designware - platdrv : Drop class based scanning to improve bootup time
firewire : fix failure path in ohci_enable_phys_dma
vmxnet3 : validate interrupt indices read on migration
TPM : tpm_nsc : Fix a double free of pdev in cleanup_nsc
hsi : nokia - modem : fix uninitialized device pointer
atl1c : return PCI_ERS_RESULT_DISCONNECT on permanent failure
clk : hisilicon : hi3660 ： Fix potential NULL dereference in hi3660_stub_clk_probe ()
[ PATCH ] e1000 : Enable polling before enabling interrupts
[ media ] coda : v4l2 - compliance fix : zero pixel format priv field
mfd : Add free max8998 -> ono irq in max8998_irq_exit ()
rpmsg : char : release allocated memory
hwmon : ( adm1025 ) Fix vrm write operation
Btrfs : add missing compression property remove in btrfs_ioctl_setflags
pinctrl : pinctrl_select_state : set the old_state back on error
mfd : arizona : Check errors from regcache_sync ()
amd - xgbe : Check for complete packet on skb allocation error
cpufreq_stats : do not remove sysfs files if frequency table is not present
[ NETFILTER ]: nfnetlink_log : fix possible NULL pointer dereference
rtc : Fix ioctl error path return
f2fs : allocate new segment aligned with sections
dm log writes : move IO accounting earlier to fix error path
kvm - irqchip : only commit route when irqchip is used
sh_eth : enable wake - on - lan for sh7734
[ media ] coda : v4l2 - compliance fix : zero pixel format priv field
[ PATCH ] Bluetooth : fix potential NULL ptr deref in dtl1_cs . c :: dtl1_hci_send_frame ()
tcp : tcp_disconnect () should clear window_clamp
usb : core : check for valid id_table when using the RefId feature
ASoC : tlv320dac33 : Avoid powering off while in BIAS_OFF
e1000e : static analysis tools complain of a possible null ptr p dereference
vmxnet3 : validate interrupt indices read on migration
pinctrl : pinctrl_select_state : set the old_state back on error
Input : wistron_btns - fix a memory leak in wb_module_init error path
rtlwifi : rtl8192cu : Fix unitialized struct
hls : fix leaking avio_opts on hls_read_header error
firewire : fix failure path in ohci_enable_phys_dma
[ PATCH ] powerpc : IOMMU : don ' t ioremap null addresses
KVM : Improve create VCPU parameter ( CVE - 2013 - 4587 )
hls : fix leaking avio_opts on hls_read_header error
xfrm : Restrict extended sequence numbers to esp
UIO : uio_pdrv : fix memory leak
mm , fs : check for fatal signals in do_generic_file_read ()
KVM : MCE : Add MCG_SER_P into KVM_MCE_CAP_SUPPORTED
kvm - irqchip : only commit route when irqchip is used
UDF : Close small mem leak in udf_find_entry ()
avformat / movenc : Check packet in mov_write_single_packet () too
[ Bluetooth ] Revert session reference counting fix
branches / zip : row_sel_sec_rec_is_for_clust_rec (): Augment the bug fix in r918 :
cifs : ensure that vol -> username is not NULL before running strlen on it
ptp_kvm : probe for kvm guest availability
mlx4_core : Fix memory leak in mlx4_enable_msi_x ()
drm / amd / display : Add same check as reset pipes for programing backend regs .
pty : fix data loss when stopped (^ S /^ Q )
brstm : fix use of uninitialized variables
omap3 : Fix booting if package is uninitialized
drm / i915 / userptr : reject zero user_size
[ SCSI ] qla2xxx : Correct IOCB queueing mechanism for ISP54XX HBAs .
mwifiex : pcie : Fix memory leak in mwifiex_pcie_alloc_cmdrsp_buf
virtio_net : zero is an invald queue_pairs number
misc : apds9802als : Fix the logic checking timeout in als_wait_for_data_ready ()
UIO : uio_pdrv : fix memory leak
wimax : i2400 : Fix memory leak in i2400m_op_rfkill_sw_toggle
ASoC : Intel : Skylake : Fix memory leak during init instance
ocfs2 : fix a memory leak in __ocfs2_move_extents ()
[ media ] xc5000 : use after free in release ()
HWPOISON : Fix page count leak in hwpoison late kill in do_swap_page
cfq - iosched : handle failure of cfq group allocation
drm / radeon / atom : workaround vbios bug in transmitter table on rs780
[ SCSI ] libfcoe : don ' t send ELS in FIP mode if no FCF selected
rtc : ac100 : Add NULL checking for devm_kzalloc call
[ SPARC ]: Fix exec failures on sun4c .
drm / i915 / skl : Restore the DDI translation tables when enabling PW1
irda : Fix missing msg_namelen update in irda_recvmsg_dgram ()
Bugfix : Free shares discarded before beginning submission
fs_enet : Fix a memory leak in fs_enet_mdio_probe
ACPI : resolve HP nx6125 S3 immediate wakeup regression
fs / 9p : destroy fid on failed remove
kvm - irqchip : only commit route when irqchip is used
ARM : zynq : initialize slcr mapping earlier
OMAPDSS : HDMI5 : disable interlace modes
mwifiex : pcie : Fix memory leak in mwifiex_pcie_alloc_cmdrsp_buf
pinctrl : s3c24xx : off by one in s3c24xx_eint_init ()
pinctrl : lpc18xx : fix schmitt trigger setup
staging : unisys : visorbus : visorchipset . c : Fix bug in parser_init_byte_stream .
UIO : uio_pdrv : fix memory leak
aoe : error printed 1 too early
net : avoid NULL deref in inet_ctl_sock_destroy ()
arm64 : Move BP hardening to check_and_switch_context
drivers : misc : pass miscdevice pointer via file private data
[ media ] coda : v4l2 - compliance fix : zero pixel format priv field
Input : amijoy - add missing platform check
mac80211 : simplify AP interface stop
drm / cirrus : Constify function pointer structs
refs : prevent accidental NULL dereference in write_pseudoref
http : Don ' t use the normal http proxy mechanism for https
x86 / smpboot : Move data structure to its primary usage scope
fscrypto : add authorization check for setting encryption policy
staging : ozwpan : Check error condition before creating endpoint .
ARM : OMAP : Use kcalloc () in omap_system_dma_probe ()
ath9k_hw : fix an off - by - one error in the PDADC boundaries calculation
Made it so cores or threads can be allocated on
UDF : Close small mem leak in udf_find_entry ()
target / iblock : Add checks for backend DIF emulation
tpm / st33zp24 / spi : Improve st33zp24_spi_evaluate_latency
x86 / smpboot : Move data structure to its primary usage scope
reset : Fix of_reset_control_get () for consistent return values
KVM : ppc : deliver INTERRUPT_FP_UNAVAIL to the guest
Drivers : hv : vmbus : Fix a bug in vmbus_open ()
[ PATCH ] KVM : Add missing ' break '
mds : handle MMonMap
[ PATCH ] ieee1394 : fix cross_bound check for null ISO packets
AFS : Fix afs_prepare_write ()
Made it so cores or threads can be allocated on
ask - password : use kill ( PID , 0 ) before querying a password
crypto : qat - update init_esram for C3xxx dev type
libsepol / cil : cil_reset_classperms_set () should not reset classpermission
ASoC : fix uninitialised variable in siu_dai . c
ALSA : Fix a Oops bug in omap soc driver .
iommu : smar : Fix return value check of create_irq ()
[ PATCH ] TPM : fix failure path leak
[ Bluetooth ] Revert session reference counting fix
branches / zip : row_sel_sec_rec_is_for_clust_rec (): Augment the bug fix in r918 :
spapr_drc : Expose ' null ' in qom - get when there is no fdt
[ ALSA ] ca0106 - Add missing sysfs device assignment
kvm : irqchip : fix memory leak
pinctrl : pinctrl_select_state : set the old_state back on error
bitops : fix test_and_change_bit ()
drivers / net / phy / mdio_bus . c : call put_device on device_register () failure
drm / cirrus : Constify function pointer structs
usb : gadget : ci13xxx_pci : guard against devices without driver_data
amd - xgbe : Check for complete packet on skb allocation error
getname_kernel () needs to make sure that -> name != -> iname in long case
staging : iio : iio_ring_rip_outer return immediately if rip_lots returns <= 0
e100 : make sure vlan support isn ' t advertised on old adapters
kvm - irqchip : only commit route when irqchip is used
ARM : 7857 / 1 : dma : imx - sdma : setup dma mask
mtd : onenand : omap2 : Remove skip initial unlocking support
hls : fix leaking avio_opts on hls_read_header error
dict - file : Updated dotlock settings .
drm : sti : Fix return value check in sti_drm_platform_probe ()
[ PATCH ] compat_sys_nfsservctl (): handle errors correctly
modules / usrloc : init _ul_sruid also in mi_child_init
staging : iio : lpc32xx_adc : Fix IRQ check
ALSA : fm801 : explicitly free IRQ line
TPM : tpm_nsc : Fix a double free of pdev in cleanup_nsc
bridge : Clear IPCB before possible entry into IP stack
powerpc / powernv : Initialize DMA for PEs
staging : unisys : Add checks for creation
KVM : PIT : control word is write - only
[ media ] coda : v4l2 - compliance fix : zero pixel format priv field
mm : numa : Do not account for a hinting fault if we raced
ALSA : usb - audio : add DSD support for new Amanero PID
[ IPV4 ] fib_trie : fix duplicated route issue
ipv6 : Export nf_ct_frag6_gather ()
netrom : fix info leak via msg_name in nr_recvmsg ()
slub : Zero initial memory segment for kmem_cache and kmem_cache_node
iwlwifi : mvm : don ' t allow sched scans without matches to be started
hls : fix leaking avio_opts on hls_read_header error
vmxnet3 : validate queues configuration read on migration
gpio : GPIO_GET_CHIPINFO_IOCTL : Fix information leak
[ POWERPC ] spufs : fix another off - by - one bug in spufs_mbox_read
Changed the pbs_tcp_timeout value when reading the protocol
ath9k : Fix a crash in 2 WIRE btcoex chipsets
mount : fix for complex automounts
[ media ] lirc : return ENOTTY when device does support ioctl
Input : amijoy - add missing platform check
[ IPV6 ] ROUTE : Prefer reachable nexthop only if the caller requests .
spi : qup : Fix return value checking for pm_runtime_get_sync ()
[ XFRM ]: xfrm_migrate () needs exporting to modules .
cifs : fix off - by - one bug in build_unc_path_to_root
e1000e : magic number cleanup - ETH_ALEN
ARM : 7857 / 1 : dma : imx - sdma : setup dma mask
Modified various APIs to match our intent .
kvm - irqchip : only commit route when irqchip is used
ixgbe : use kzalloc for allocating one thing
drm / i915 : fix integer overflow in i915_gem_execbuffer2 ()
TG3 : limit reaches - 1
iio : adc : meson - saradc : unlock on error in meson_sar_adc_lock ()
Input : amijoy - add missing platform check
Input : gpio_keys - fix a memory leak
sched : Fix information leak in sys_sched_getattr ()
extcon : Fix return value in extcon - class . c
drm / nouveau / devinit / nv04 : change owner to int
locale : make sure that l is freed
ASoC : Intel : Skylake : Fix memory leak during init instance
PCI : Fix devm_pci_alloc_host_bridge () memory leak
blkcg : don ' t call into policy draining if root_blkg is already gone
b43 : Force - wake queues on init
mac80211 : add status_driver_data array to ieee80211_tx_info
vmxnet3 : validate interrupt indices read on migration
added a security check in the http router
drm / i915 / skl : Ensure pipes with changed wms get added to the state
UDF : Close small mem leak in udf_find_entry ()
Modified various APIs to match our intent .
mmc : sdhci - acpi : Handle return value of platform_get_irq
iio : buffer : Free active scan mask in iio_disable_all_buffers ()
virtio_net : enable big packets for large MTU values
[ media ] s5p - fimc : Fix return value of __fimc_md_create_flite_source_links ()
gpio - mcp23s08 : correctly handling failed allocation
ARM : OMAP : Use kcalloc () in omap_system_dma_probe ()
[ SCTP ]: Validate the parameter length in HB - ACK chunk .
[ media ] au8522_common : add missing MODULE_LICENSE
ath6kl : add missing of_node_put
UDF : Close small mem leak in udf_find_entry ()
[ PATCH ] ieee1394 : fix cross_bound check for null ISO packets
ipv6 - multicast : Fix memory leak in input path .
kvm - irqchip : only commit route when irqchip is used
ARM : zynq : initialize slcr mapping earlier
ASoC : Intel : Skylake : Fix memory leak during init instance
ACPICA : Fix for Linux source translation
[ SPARC64 ]: Use alloc_pci_dev () in PCI bus probes .
ext4 : fix memory leak in xattr
ath10k : fix memory leak
avcodec / mpegaudiodec_template : Make l3_unscale () work with e = 0
tcp : remove unnecessary tcp_sk assignment .
mwifiex : pcie : Fix memory leak in mwifiex_pcie_alloc_cmdrsp_buf
[ Bluetooth ] Revert session reference counting fix
modules / usrloc : init _ul_sruid also in mi_child_init
ravb : Add missing free_irq () call to ravb_close ()
floppy : silence warning during disk test
ibmvnic : Fix NAPI structures memory leak
Hardware : Fix possible recursive locking in hwregs . c
fcoe : ensure that skb placed on the fip_recv_list are unshared
[ ALSA ] Add missing compat ioctls for ALSA control API
Bugfix : Free shares discarded before beginning submission
ACPICA : Fix for Linux source translation
memcg : fix leak of accounting at failure path of hugepage collapsing
bitmap . h : Don ' t include qemu - common . h
avformat / oggparseogm : check input size before reading t
HID : hidraw , fix a NULL pointer dereference in hidraw_ioctl
ov7670 : clean up ov7670_read semantics
Input : amijoy - add missing platform check
hsi : nokia - modem : fix uninitialized device pointer
tg3 : remove open - coded skb_cow_head .
staging : comedi : ni_at_ao : fix namespace clash of ' RSI ' define
Fixed potential buffer overflow in ` ProcessRadioRxDone `
GFS2 : quotas not being refreshed in gfs2_adjust_quota
drm / i915 : Fix missing return warning for ! CONFIG_DEBUGFS
vmxnet3 : validate interrupt indices read on migration
x86 : add dummy acpi_numa_processor_affinity_init () implementation on 32 - bit
Fix a read past end of buffer crash in the mp3 probe
irqchip / gic - v3 : Fix out - of - bound access in gic_set_affinity
KVM : ARM : VGIC : Optimize the vGIC vgic_update_irq_pending function .
USB : serial : console : fix use - after - free on disconnect
[ Bluetooth ] Revert session reference counting fix
Staging : Fix leak in drivers / staging / at76_usb . c
drm / nouveau / i2c / gp104 : initial support
misc : mei : fix stalled read
vnc : fix qemu crash when not configure vnc option
iio : chemical : ccs811 : Corrected firmware boot / application mode transition
spi : qup : Fix return value checking for pm_runtime_get_sync ()
vmxnet3 : validate interrupt indices read on migration
mkv : mark corrupted packets and return them
hwmon : ( adm1025 ) Fix vrm write operation
[ SCSI ] scsi_lib : Fix bug in completion of bidi commands
drbd : fix possible access after free
dccp : Empty the write queue when disconnecting
[ SCSI ] scsi_transport_iscsi : make priv_sess file writeable only by root
fs : make inode_to_bdi () handle NULL inode
ath10k : enable Adaptive Noise Immunity ( ANI ) by default
reparent_leader : check EXIT_DEAD instead of task_detached ()
misc : fastrpc : prevent memory leak in fastrpc_dma_buf_attach
ARM : 7803 / 1 : Fix deadlock scenario with smp_send_stop ()
powerpc / pseries : Fix possible leaked device node reference
drm / cirrus : Constify function pointer structs
avcodec / hevc : make * ps_id unsigned
kvm - irqchip : only commit route when irqchip is used
branches / zip : row_sel_sec_rec_is_for_clust_rec (): Augment the bug fix in r918 :
cw1200 : Fix cw1200_debug_link_id
pinctrl : pinctrl_select_state : set the old_state back on error
Btrfs : fix incorrect space accounting after failure to insert inline extent
ui : Make qemu_default_pixman_format () return 0 on unsupported formats
ath6kl : add missing of_node_put
git - compat - util : make UNLEAK less error - prone
Bugfix : free adhoc string elist element when removing it from list
Input : amijoy - add missing platform check
vhost - net : fix vq_memory_access_ok error checking
qemu - io : fix a memory leak
[ lite ] Add validation check for dilation height / width to be positive integers .
[ lite ] Add validation check for dilation height / width to be positive integers .
ACPI : check acpi_disabled in acpi_table_parse () and acpi_table_parse_entries ()
exit - on - idle : only exit if actually idle
fix buffer overrun in gf_bin128_parse
* mini - s390 [ x ]. c ( is_regsize_var ): Support PTR / FNPTR too .
ALSA : hda - fix the mic mute led problem for Latitude E5550
arm64 : drop unnecessary cache + tlb maintenance
kvm - irqchip : only commit route when irqchip is used
