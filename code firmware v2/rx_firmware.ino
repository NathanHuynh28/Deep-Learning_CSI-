/* ========================================================
   CSI DATASET RX FIRMWARE V4.7 (MAC-SYNC & CSI DIAGNOSTIC)
   Board: ESP32-S3 receiver
   Purpose: 100Hz Rock-solid. 64 Subcarriers = 128 I/Q values.
   ======================================================== */

#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

#define WIFI_CHANNEL       1
#define RX_ID              3   // Flash RX1=1, RX2=2, RX3=3
#define EXPECTED_TX_ID     1
#define PAYLOAD_MAGIC      0xAA
#define PAYLOAD_VERSION    2
#define EXPECTED_CSI_LEN   128 // TRỞ LẠI BẢN CHẤT GỐC: 128 số I/Q
#define SERIAL_BAUD        921600 
#define OUTPUT_QUEUE_SIZE  32
#define CSI_LINE_BUF_SIZE  1200

#define USE_TX_MAC_FILTER  1
uint8_t TX_MAC_FILTER[6] = {0xAC, 0xA7, 0x04, 0x1D, 0xA0, 0x30};

typedef struct __attribute__((packed)) {
  uint8_t magic;
  uint8_t version;
  uint8_t tx_id;
  uint8_t channel;
  uint32_t seq;
  uint32_t tx_time_ms;
  uint16_t checksum;
} TxPacket;

static_assert(sizeof(TxPacket) == 14, "TxPacket must stay packed at 14 bytes");

typedef struct {
  uint32_t mac_time;
  int8_t rssi;
  uint8_t channel;
  uint8_t rate;
  uint8_t cwb;
  uint8_t rx_state;
  uint8_t first_word_invalid;
  uint8_t csi_len;
  uint8_t src_mac[6];
  int8_t csi[EXPECTED_CSI_LEN];
  bool ready;
} PhyData;

typedef struct {
  uint8_t rx_id;
  uint8_t tx_id;
  uint32_t seq;
  uint32_t mac_time;
  int8_t rssi;
  uint8_t channel;
  uint8_t rate;
  uint8_t cwb;
  uint8_t rx_state;
  uint8_t first_word_invalid;
  uint8_t csi_len;
  uint8_t src_mac[6];
  int8_t csi[EXPECTED_CSI_LEN];
} OutputPacket;

PhyData latest_phy;
portMUX_TYPE phy_mux = portMUX_INITIALIZER_UNLOCKED;
QueueHandle_t outputQueue;

uint32_t csi_seen_count = 0;
uint32_t csi_callback_count = 0;
uint32_t espnow_recv_count = 0;
uint32_t espnow_valid_count = 0;
uint32_t seq_gap_count = 0;
uint32_t last_seq = 0;
bool has_last_seq = false;
uint32_t dropped_csi_null_count = 0;
uint32_t dropped_csi_mac_mismatch_count = 0;
uint32_t valid_packet_count = 0;
uint32_t dropped_no_csi_count = 0;
uint32_t dropped_bad_payload_count = 0;
uint32_t dropped_bad_tx_id_count = 0;
uint32_t dropped_bad_channel_count = 0;
uint32_t dropped_bad_checksum_count = 0;
uint32_t dropped_bad_csi_len_count = 0;
uint32_t dropped_mac_mismatch_count = 0;
uint32_t dropped_output_busy_count = 0;
uint32_t last_csi_len = 0;
uint32_t last_stats_ms = 0;

uint16_t make_checksum(const TxPacket &packet) {
  uint16_t value = 0;
  value ^= packet.magic;
  value ^= ((uint16_t)packet.version << 8) | packet.tx_id;
  value ^= packet.channel;
  value ^= (uint16_t)(packet.seq & 0xFFFF);
  value ^= (uint16_t)((packet.seq >> 16) & 0xFFFF);
  value ^= (uint16_t)(packet.tx_time_ms & 0xFFFF);
  value ^= (uint16_t)((packet.tx_time_ms >> 16) & 0xFFFF);
  return value;
}

bool mac_matches_filter(const uint8_t *mac) {
#if USE_TX_MAC_FILTER
  return memcmp(mac, TX_MAC_FILTER, 6) == 0;
#else
  (void)mac;
  return true;
#endif
}

size_t append_char(char *buffer, size_t pos, char value) {
  if (pos < CSI_LINE_BUF_SIZE - 1) buffer[pos++] = value; return pos;
}
size_t append_str(char *buffer, size_t pos, const char *value) {
  while (*value && pos < CSI_LINE_BUF_SIZE - 1) buffer[pos++] = *value++; return pos;
}
size_t append_uint32(char *buffer, size_t pos, uint32_t value) {
  char digits[10]; int count = 0;
  do { digits[count++] = (char)('0' + (value % 10)); value /= 10; } while (value > 0 && count < 10);
  while (count > 0 && pos < CSI_LINE_BUF_SIZE - 1) buffer[pos++] = digits[--count];
  return pos;
}
size_t append_int32(char *buffer, size_t pos, int32_t value) {
  if (value < 0) { pos = append_char(buffer, pos, '-'); value = -value; }
  return append_uint32(buffer, pos, (uint32_t)value);
}
size_t append_hex_byte(char *buffer, size_t pos, uint8_t value) {
  const char hex[] = "0123456789ABCDEF";
  pos = append_char(buffer, pos, hex[(value >> 4) & 0x0F]); return append_char(buffer, pos, hex[value & 0x0F]);
}
size_t append_mac_text(char *buffer, size_t pos, const uint8_t *mac) {
  for (int i = 0; i < 6; i++) { if (i > 0) pos = append_char(buffer, pos, ':'); pos = append_hex_byte(buffer, pos, mac[i]); }
  return pos;
}

void print_csi_packet(const OutputPacket &packet) {
  // KHỐI 1: IN HEADER ĐỂ GIẢM TẢI BỘ ĐỆM
  static char header_line[200]; 
  size_t pos = 0;

  pos = append_str(header_line, pos, "CSI_V2,");
  pos = append_uint32(header_line, pos, packet.rx_id); pos = append_char(header_line, pos, ',');
  pos = append_uint32(header_line, pos, packet.tx_id); pos = append_char(header_line, pos, ',');
  pos = append_uint32(header_line, pos, packet.seq); pos = append_char(header_line, pos, ',');
  pos = append_uint32(header_line, pos, packet.mac_time); pos = append_char(header_line, pos, ',');
  pos = append_int32(header_line, pos, packet.rssi); pos = append_char(header_line, pos, ',');
  pos = append_uint32(header_line, pos, packet.channel); pos = append_char(header_line, pos, ',');
  pos = append_uint32(header_line, pos, packet.rate); pos = append_char(header_line, pos, ',');
  pos = append_uint32(header_line, pos, packet.cwb); pos = append_char(header_line, pos, ',');
  pos = append_uint32(header_line, pos, packet.rx_state); pos = append_char(header_line, pos, ',');
  pos = append_uint32(header_line, pos, packet.first_word_invalid); pos = append_char(header_line, pos, ',');
  pos = append_uint32(header_line, pos, EXPECTED_CSI_LEN); pos = append_char(header_line, pos, ',');
  pos = append_mac_text(header_line, pos, packet.src_mac);
  pos = append_char(header_line, pos, ','); 
  
  Serial.write((const uint8_t *)header_line, pos);

  // KHỐI 2: IN NGUYÊN BẢN 128 SỐ I/Q BẰNG VÒNG LẶP SIÊU NHANH
  static char csi_line[CSI_LINE_BUF_SIZE];
  size_t csi_pos = 0;
  
  for (int i = 0; i < EXPECTED_CSI_LEN; i++) {
    csi_pos = append_int32(csi_line, csi_pos, packet.csi[i]);
    if (i < EXPECTED_CSI_LEN - 1) {
        csi_pos = append_char(csi_line, csi_pos, ',');
    }
  }
  csi_pos = append_char(csi_line, csi_pos, '\n');
  
  Serial.write((const uint8_t *)csi_line, csi_pos);
}

void csi_callback(void *ctx, wifi_csi_info_t *info) {
  (void)ctx;
  csi_callback_count++;
  if (info == NULL || info->buf == NULL) {
    dropped_csi_null_count++;
    return;
  }
  last_csi_len = info->len;
  
  // KIỂM TRA ĐỘ DÀI: Chấp nhận mọi độ dài >= 128 (để tương thích ESP32-S3)
  if (info->len < EXPECTED_CSI_LEN) {
    dropped_bad_csi_len_count++;
    return;
  }
  if (!mac_matches_filter(info->mac)) {
    dropped_csi_mac_mismatch_count++;
    return;
  }

  portENTER_CRITICAL(&phy_mux);
  // CHỈ DUMP ĐÚNG 128 BYTE ĐẦU TIÊN TỪ RAM (64 Subcarriers I/Q)
  memcpy((void *)latest_phy.csi, info->buf, EXPECTED_CSI_LEN);
  memcpy((void *)latest_phy.src_mac, info->mac, 6);
  latest_phy.mac_time = info->rx_ctrl.timestamp;
  latest_phy.rssi = info->rx_ctrl.rssi;
  latest_phy.channel = info->rx_ctrl.channel;
  latest_phy.rate = info->rx_ctrl.rate;
  latest_phy.cwb = info->rx_ctrl.cwb;
  latest_phy.rx_state = info->rx_ctrl.rx_state;
  latest_phy.first_word_invalid = info->first_word_invalid ? 1 : 0;
  latest_phy.csi_len = EXPECTED_CSI_LEN;
  latest_phy.ready = true;
  csi_seen_count++;
  portEXIT_CRITICAL(&phy_mux);
}

void handle_recv_packet(const uint8_t *mac, const uint8_t *incoming_data, int len) {
  espnow_recv_count++;
  if (!mac_matches_filter(mac)) { dropped_mac_mismatch_count++; return; }
  if (len != sizeof(TxPacket)) { dropped_bad_payload_count++; return; }

  TxPacket packet;
  memcpy(&packet, incoming_data, sizeof(TxPacket));
  if (packet.magic != PAYLOAD_MAGIC || packet.version != PAYLOAD_VERSION) { dropped_bad_payload_count++; return; }
  if (packet.tx_id != EXPECTED_TX_ID) { dropped_bad_tx_id_count++; return; }
  if (packet.channel != WIFI_CHANNEL) { dropped_bad_channel_count++; return; }
  if (packet.checksum != make_checksum(packet)) { dropped_bad_checksum_count++; return; }
  espnow_valid_count++;
  if (has_last_seq && packet.seq > last_seq + 1) {
    seq_gap_count += packet.seq - last_seq - 1;
  }
  last_seq = packet.seq;
  has_last_seq = true;

  OutputPacket queued;
  portENTER_CRITICAL(&phy_mux);
  if (!latest_phy.ready) {
    portEXIT_CRITICAL(&phy_mux);
    dropped_no_csi_count++;
    return;
  }
  if (memcmp(mac, latest_phy.src_mac, 6) != 0) {
    latest_phy.ready = false;
    portEXIT_CRITICAL(&phy_mux);
    dropped_mac_mismatch_count++;
    return;
  }

  queued.rx_id = RX_ID;
  queued.tx_id = packet.tx_id;
  queued.seq = packet.seq;
  queued.mac_time = latest_phy.mac_time;
  queued.rssi = latest_phy.rssi;
  queued.channel = latest_phy.channel;
  queued.rate = latest_phy.rate;
  queued.cwb = latest_phy.cwb;
  queued.rx_state = latest_phy.rx_state;
  queued.first_word_invalid = latest_phy.first_word_invalid;
  queued.csi_len = latest_phy.csi_len;
  memcpy(queued.src_mac, latest_phy.src_mac, 6);
  memcpy(queued.csi, latest_phy.csi, EXPECTED_CSI_LEN);
  latest_phy.ready = false;
  portEXIT_CRITICAL(&phy_mux);

  if (xQueueSend(outputQueue, &queued, 0) != pdTRUE) {
    dropped_output_busy_count++;
  }
}

#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
void on_data_recv(const esp_now_recv_info_t *recv_info, const uint8_t *incoming_data, int len) {
  handle_recv_packet(recv_info->src_addr, incoming_data, len);
}
#else
void on_data_recv(const uint8_t *mac, const uint8_t *incoming_data, int len) {
  handle_recv_packet(mac, incoming_data, len);
}
#endif

void setup_wifi() {
  WiFi.mode(WIFI_STA);
  WiFi.disconnect(false, true);
  delay(100);
  esp_wifi_set_ps(WIFI_PS_NONE);
  esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
  esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_11N);
  esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20);
  esp_wifi_set_promiscuous(true); 
}

void setup_csi() {
  wifi_csi_config_t csi_config = {
    .lltf_en = false,
    .htltf_en = true,
    .stbc_htltf2_en = false,
    .ltf_merge_en = false,
    .channel_filter_en = false,
    .manu_scale = false,
    .shift = false,
  };
  esp_err_t config_result = esp_wifi_set_csi_config(&csi_config);
  esp_err_t callback_result = esp_wifi_set_csi_rx_cb(csi_callback, NULL);
  esp_err_t enable_result = esp_wifi_set_csi(true);
  Serial.printf("CONFIG,csi_promiscuous,1,lltf,0,htltf,1,stbc_htltf2,0,ltf_merge,0,channel_filter,0,set_config,%d,set_cb,%d,set_enable,%d\n",
                config_result, callback_result, enable_result);
}

void setup() {
  Serial.setTxBufferSize(16384);
  Serial.begin(SERIAL_BAUD);
  delay(500);

  outputQueue = xQueueCreate(OUTPUT_QUEUE_SIZE, sizeof(OutputPacket));
  if (outputQueue == NULL) {
    Serial.println("ERR,output_queue_create_failed");
    while (true) delay(1000);
  }

  setup_wifi();
  if (esp_now_init() != ESP_OK) {
    Serial.println("ERR,esp_now_init_failed");
    while (true) delay(1000);
  }
  esp_now_register_recv_cb(on_data_recv);
  setup_csi();

  Serial.println("RX_READY_V4_7_CSI_DIAG_128");
  Serial.printf("CONFIG,rx_id,%u,expected_tx_id,%u,channel,%u,csi_len,%u\n",
                RX_ID, EXPECTED_TX_ID, WIFI_CHANNEL, EXPECTED_CSI_LEN);
  last_stats_ms = millis();
}

void loop() {
  OutputPacket packet;
  if (xQueueReceive(outputQueue, &packet, 0) == pdTRUE) {
    valid_packet_count++;
    print_csi_packet(packet);
  }

  uint32_t now_ms = millis();
  if (now_ms - last_stats_ms >= 1000) {
    last_stats_ms = now_ms;
    Serial.printf("STAT,rx_id,%u,csi_cb,%lu,csi_seen,%lu,valid,%lu,espnow_recv,%lu,espnow_valid,%lu,seq_gap,%lu,last_seq,%lu,csi_null,%lu,csi_mac_mismatch,%lu,no_csi,%lu,output_busy,%lu,bad_payload,%lu,bad_tx_id,%lu,bad_channel,%lu,bad_checksum,%lu,bad_csi_len,%lu,mac_mismatch,%lu,last_csi_len,%lu,queue,%u,channel,%u\n",
                  RX_ID,
                  (unsigned long)csi_callback_count,
                  (unsigned long)csi_seen_count,
                  (unsigned long)valid_packet_count,
                  (unsigned long)espnow_recv_count,
                  (unsigned long)espnow_valid_count,
                  (unsigned long)seq_gap_count,
                  (unsigned long)last_seq,
                  (unsigned long)dropped_csi_null_count,
                  (unsigned long)dropped_csi_mac_mismatch_count,
                  (unsigned long)dropped_no_csi_count,
                  (unsigned long)dropped_output_busy_count,
                  (unsigned long)dropped_bad_payload_count,
                  (unsigned long)dropped_bad_tx_id_count,
                  (unsigned long)dropped_bad_channel_count,
                  (unsigned long)dropped_bad_checksum_count,
                  (unsigned long)dropped_bad_csi_len_count,
                  (unsigned long)dropped_mac_mismatch_count,
                  (unsigned long)last_csi_len,
                  (unsigned int)uxQueueMessagesWaiting(outputQueue),
                  WIFI_CHANNEL);
  }
}
