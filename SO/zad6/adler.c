#include <minix/drivers.h>
#include <minix/chardriver.h>
#include <stdio.h>
#include <stdlib.h>
#include <minix/ds.h>

#define BSIZE          5550
#define MOD_ADLER      65521
#define CONST_ADLER    65536
#define CONTRL_SUM_LEN 8

static uint32_t a, b;
static char control_sum[CONTRL_SUM_LEN+1];

static ssize_t adler_read(devminor_t minor, u64_t position, endpoint_t endpt,
    cp_grant_id_t grant, size_t size, int flags, cdev_id_t id);

static ssize_t adler_write(devminor_t minor, u64_t position, endpoint_t endpt,
    cp_grant_id_t grant, size_t size, int flags, cdev_id_t id);

static void sef_local_startup(void);
static int sef_cb_init(int type, sef_init_info_t *info);
static int sef_cb_lu_state_save(int);
static int lu_state_restore(void);

static struct chardriver adler_tab =
{
    .cdr_read	= adler_read,
    .cdr_write = adler_write,
};

void update_adler(uint8_t *data, size_t len) {
  do {
    a += *data++;
    b += a;
  } while (--len);
  a = (a & 0xffff) + (a >> 16) * (CONST_ADLER-MOD_ADLER);
  b = (b & 0xffff) + (b >> 16) * (CONST_ADLER-MOD_ADLER);
  if (a >= MOD_ADLER)
    a -= MOD_ADLER;
  b = (b & 0xffff) + (b >> 16) * (CONST_ADLER-MOD_ADLER);
  if (b >= MOD_ADLER)
    b -= MOD_ADLER;
  sprintf(control_sum, "%08x", (b << 16) | a);
}

void reset_control_sum() {
  a = 1;
  b = 0;
  sprintf(control_sum, "%08x", 1);
}

static ssize_t adler_read(devminor_t UNUSED(minor), u64_t position,
    endpoint_t endpt, cp_grant_id_t grant, size_t size, int UNUSED(flags),
    cdev_id_t UNUSED(id))
{
    if (size < CONTRL_SUM_LEN)
      return EINVAL;

    u64_t dev_size;
    char *ptr;
    int ret;
    char *buf = control_sum;
    dev_size = (u64_t) strlen(buf);

    if (position >= dev_size) return 0;
    if (position + size > dev_size)
        size = (size_t)(dev_size - position);

    ptr = buf + (size_t)position;
    if ((ret = sys_safecopyto(endpt, grant, 0, (vir_bytes) ptr, size)) != OK)
        return ret;

    reset_control_sum();
    return size;
}

static ssize_t adler_write(devminor_t UNUSED(minor), u64_t position,
    endpoint_t endpt, cp_grant_id_t grant, size_t size, int UNUSED(flags),
    cdev_id_t UNUSED(id))
{
  char data[BSIZE];
  size_t bytes_left = size;
  ssize_t ret;

  while(bytes_left > 0) {
    size_t bytes_to_write = (bytes_left > BSIZE) ? BSIZE : bytes_left;
    if ((ret = sys_safecopyfrom(endpt, grant, size-bytes_left, (vir_bytes) data, bytes_to_write)) != OK)
      return ret;
    update_adler(data, bytes_to_write);
    bytes_left -= bytes_to_write;
  }
  return size;
}

static int sef_cb_lu_state_save(int UNUSED(state)) {
  ds_publish_u32("a", a, DSF_OVERWRITE);
  ds_publish_u32("b", b, DSF_OVERWRITE);
  return OK;
}

static int lu_state_restore() {
  ds_retrieve_u32("a", &a);
  ds_retrieve_u32("b", &b);
  ds_delete_u32("a");
  ds_delete_u32("b");
  sprintf(control_sum, "%08x", (b << 16) | a);
  return OK;
}

static void sef_local_startup() {
    sef_setcb_init_fresh(sef_cb_init);
    sef_setcb_init_lu(sef_cb_init);
    sef_setcb_init_restart(sef_cb_init);

    sef_setcb_lu_prepare(sef_cb_lu_prepare_always_ready);
    sef_setcb_lu_state_isvalid(sef_cb_lu_state_isvalid_standard);
    sef_setcb_lu_state_save(sef_cb_lu_state_save);

    sef_startup();
}

static int sef_cb_init(int type, sef_init_info_t *UNUSED(info)) {
    reset_control_sum();
    if (type == SEF_INIT_LU)
      lu_state_restore();
    else
      chardriver_announce();
    return OK;
}

int main(void) {
    sef_local_startup();
    chardriver_task(&adler_tab);
    return OK;
}
