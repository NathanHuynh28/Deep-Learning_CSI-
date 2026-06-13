// /* ========================================================
//    CSI DATASET TX FIRMWARE V2
//    Board: ESP32-S3 transmitter
//    Purpose: send deterministic ESP-NOW packets for CSI capture
//    Flash with Arduino IDE.
//    ======================================================== */

// #include <WiFi.h>
// #include <esp_now.h>
// #include <esp_wifi.h>

// // ===== EDIT BEFORE EACH COLLECTION SESSION IF NEEDED =====
// #define WIFI_CHANNEL       1
// #define TX_ID              1
// #define PAYLOAD_MAGIC      0xAA
// #define PAYLOAD_VERSION    2
// #define SEND_INTERVAL_US   10000UL  // 100 Hz
// #define SERIAL_BAUD        921600

// // Keep HT20 for 64 subcarriers per RX (128 raw I/Q values).
// #define TX_POWER_LEVEL     WIFI_POWER_19_5dBm

// // ESP-NOW PHY options for real-room CSI collection:
// // - WIFI_PHY_RATE_1M_L: legacy 1 Mbps; too little airtime margin for stable 3-RX 100 Hz CSI.
// // - WIFI_PHY_RATE_MCS0_LGI: HT20 baseline; primary 3-RX 100 Hz setting.
// // - WIFI_PHY_RATE_MCS1_LGI: still robust, slightly less airtime.
// // - WIFI_PHY_RATE_MCS2_LGI: middle ground for clean rooms.
// // - WIFI_PHY_RATE_MCS4_LGI: faster airtime but needs strong, clean links.
// // Change one line below, flash TX again, then run controller preflight.
// #define ESPNOW_PHY_RATE    WIFI_PHY_RATE_MCS0_LGI

// typedef struct __attribute__((packed)) {
//   uint8_t magic;
//   uint8_t version;
//   uint8_t tx_id;
//   uint8_t channel;
//   uint32_t seq;
//   uint32_t tx_time_ms;
//   uint16_t checksum;
// } TxPacket;

// static_assert(sizeof(TxPacket) == 14, "TxPacket must stay packed at 14 bytes");

// TxPacket tx_packet;
// uint8_t broadcast_addr[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// volatile uint32_t send_ok_count = 0;
// volatile uint32_t send_fail_count = 0;
// volatile bool send_in_flight = false;
// uint32_t send_attempt_count = 0;
// uint32_t send_busy_skip_count = 0;
// uint32_t max_late_us = 0;
// uint32_t last_stats_ms = 0;
// uint32_t last_stats_seq = 0;
// uint32_t next_send_us = 0;

// uint16_t make_checksum(const TxPacket &packet) {
//   uint16_t value = 0;
//   value ^= packet.magic;
//   value ^= ((uint16_t)packet.version << 8) | packet.tx_id;
//   value ^= packet.channel;
//   value ^= (uint16_t)(packet.seq & 0xFFFF);
//   value ^= (uint16_t)((packet.seq >> 16) & 0xFFFF);
//   value ^= (uint16_t)(packet.tx_time_ms & 0xFFFF);
//   value ^= (uint16_t)((packet.tx_time_ms >> 16) & 0xFFFF);
//   return value;
// }

// void handle_send_result(esp_now_send_status_t status) {
//   if (status == ESP_NOW_SEND_SUCCESS) {
//     send_ok_count++;
//   } else {
//     send_fail_count++;
//   }
//   send_in_flight = false;
// }

// #if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
// void on_data_sent(const wifi_tx_info_t *tx_info, esp_now_send_status_t status) {
//   (void)tx_info;
//   handle_send_result(status);
// }
// #else
// void on_data_sent(const uint8_t *mac_addr, esp_now_send_status_t status) {
//   (void)mac_addr;
//   handle_send_result(status);
// }
// #endif

// void setup_wifi() {
//   WiFi.mode(WIFI_STA);
//   WiFi.disconnect(false, true);
//   delay(100);

//   esp_wifi_set_ps(WIFI_PS_NONE);
//   esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
//   esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_11N);
//   esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20);
//   esp_wifi_set_promiscuous(false);
//   WiFi.setTxPower(TX_POWER_LEVEL);

//   // Fixed ESP-NOW rate makes preflight comparisons repeatable across sessions.
//   esp_wifi_config_espnow_rate(WIFI_IF_STA, ESPNOW_PHY_RATE);
// }

// void setup_esp_now() {
//   if (esp_now_init() != ESP_OK) {
//     Serial.println("ERR,esp_now_init_failed");
//     while (true) delay(1000);
//   }

//   esp_now_register_send_cb(on_data_sent);

//   esp_now_peer_info_t peer_info = {};
//   memcpy(peer_info.peer_addr, broadcast_addr, 6);
//   peer_info.channel = WIFI_CHANNEL;
//   peer_info.ifidx = WIFI_IF_STA;
//   peer_info.encrypt = false;

//   esp_err_t add_result = esp_now_add_peer(&peer_info);
//   if (add_result != ESP_OK && add_result != ESP_ERR_ESPNOW_EXIST) {
//     Serial.printf("ERR,esp_now_add_peer_failed,%d\n", add_result);
//     while (true) delay(1000);
//   }
// }

// void setup() {
//   Serial.begin(SERIAL_BAUD);
//   delay(500);

//   setup_wifi();
//   setup_esp_now();

//   tx_packet.magic = PAYLOAD_MAGIC;
//   tx_packet.version = PAYLOAD_VERSION;
//   tx_packet.tx_id = TX_ID;
//   tx_packet.channel = WIFI_CHANNEL;
//   tx_packet.seq = 0;
//   tx_packet.tx_time_ms = 0;
//   tx_packet.checksum = make_checksum(tx_packet);

//   Serial.println("TX_READY_V2");
//   Serial.printf("CONFIG,tx_id,%u,channel,%u,interval_us,%lu,payload_size,%u\n",
//                 TX_ID, WIFI_CHANNEL, SEND_INTERVAL_US, (unsigned int)sizeof(TxPacket));
//   Serial.print("TX_MAC,");
//   Serial.println(WiFi.macAddress());

//   next_send_us = micros();
//   last_stats_ms = millis();
//   last_stats_seq = tx_packet.seq;
// }

// void loop() {
//   uint32_t now_us = micros();
//   if ((int32_t)(now_us - next_send_us) >= 0) {
//     uint32_t late_us = now_us - next_send_us;
//     if (late_us > max_late_us) max_late_us = late_us;
//     next_send_us += SEND_INTERVAL_US;

//     if (send_in_flight) {
//       send_busy_skip_count++;
//       return;
//     }

//     tx_packet.seq++;
//     tx_packet.tx_time_ms = millis();
//     tx_packet.checksum = make_checksum(tx_packet);
//     send_attempt_count++;

//     esp_err_t result = esp_now_send(broadcast_addr, (uint8_t *)&tx_packet, sizeof(tx_packet));
//     if (result != ESP_OK) {
//       send_fail_count++;
//       send_in_flight = false;
//     } else {
//       send_in_flight = true;
//     }
//   }

//   uint32_t now_ms = millis();
//   if (now_ms - last_stats_ms >= 1000) {
//     uint32_t elapsed_ms = now_ms - last_stats_ms;
//     uint32_t seq_delta = tx_packet.seq - last_stats_seq;
//     uint32_t send_rate_hz = (seq_delta * 1000UL) / max(elapsed_ms, 1UL);
//     last_stats_ms = now_ms;
//     last_stats_seq = tx_packet.seq;
//     uint32_t max_late_snapshot = max_late_us;
//     max_late_us = 0;
//     Serial.printf("STAT,seq,%lu,attempt,%lu,ok,%lu,fail,%lu,busy_skip,%lu,send_rate_hz,%lu,max_late_us,%lu,interval_us,%lu,channel,%u\n",
//                   (unsigned long)tx_packet.seq,
//                   (unsigned long)send_attempt_count,
//                   (unsigned long)send_ok_count,
//                   (unsigned long)send_fail_count,
//                   (unsigned long)send_busy_skip_count,
//                   (unsigned long)send_rate_hz,
//                   (unsigned long)max_late_snapshot,
//                   (unsigned long)SEND_INTERVAL_US,
//                   WIFI_CHANNEL);
//   }
// }


#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <esp_netif.h> // Sửa lỗi thiếu định nghĩa ESP_IF_WIFI_STA

#define WIFI_CHANNEL       1
#define TX_ID              1
#define PAYLOAD_MAGIC      0xAA
#define PAYLOAD_VERSION    2
#define SEND_INTERVAL_US   10000UL  // Cố định 100 Hz
#define SERIAL_BAUD        921600

#define TX_POWER_LEVEL     WIFI_POWER_19_5dBm
#define ESPNOW_PHY_RATE    WIFI_PHY_RATE_MCS0_LGI

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

TxPacket tx_packet;
uint8_t broadcast_addr[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

volatile uint32_t send_ok_count = 0;
volatile uint32_t send_fail_count = 0;
uint32_t send_attempt_count = 0;
uint32_t last_stats_ms = 0;
uint32_t last_stats_seq = 0;
uint32_t next_send_us = 0;

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

// Sửa lỗi Callback bị đổi cấu trúc trên Arduino Core 3.x
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
void on_data_sent(const wifi_tx_info_t *tx_info, esp_now_send_status_t status) {
  (void)tx_info;
  if (status == ESP_NOW_SEND_SUCCESS) {
    send_ok_count++;
  } else {
    send_fail_count++;
  }
}
#else
void on_data_sent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  (void)mac_addr;
  if (status == ESP_NOW_SEND_SUCCESS) {
    send_ok_count++;
  } else {
    send_fail_count++;
  }
}
#endif

void setup_wifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(); // Gọi begin để tránh lỗi STA not started
  WiFi.disconnect();
  delay(100);
  esp_wifi_set_ps(WIFI_PS_NONE);
  esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
  esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_11N);
  esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20);
  esp_wifi_set_promiscuous(false);
  WiFi.setTxPower(TX_POWER_LEVEL);
  esp_wifi_config_espnow_rate(WIFI_IF_STA, ESPNOW_PHY_RATE);
}

void setup_esp_now() {
  if (esp_now_init() != ESP_OK) while (true) delay(1000);
  
  esp_now_register_send_cb(on_data_sent);
  
  esp_now_peer_info_t peer_info = {};
  memcpy(peer_info.peer_addr, broadcast_addr, 6);
  peer_info.channel = WIFI_CHANNEL;
  peer_info.ifidx = WIFI_IF_STA;
  peer_info.encrypt = false;
  esp_now_add_peer(&peer_info);
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(500);
  setup_wifi();
  setup_esp_now();

  tx_packet.magic = PAYLOAD_MAGIC;
  tx_packet.version = PAYLOAD_VERSION;
  tx_packet.tx_id = TX_ID;
  tx_packet.channel = WIFI_CHANNEL;
  tx_packet.seq = 0;
  
  Serial.println("TX_READY_V3_1_FIRE_AND_FORGET");
  next_send_us = micros();
  last_stats_ms = millis();
}

void loop() {
  uint32_t now_us = micros();
  // KIỂM SOÁT THỜI GIAN TUYỆT ĐỐI (KHÔNG CÓ ĐIỀU KIỆN RÀNG BUỘC)
  if ((int32_t)(now_us - next_send_us) >= 0) {
    next_send_us += SEND_INTERVAL_US;

    tx_packet.seq++;
    tx_packet.tx_time_ms = millis();
    tx_packet.checksum = make_checksum(tx_packet);
    send_attempt_count++;

    // Bơm thẳng vào Queue của MAC, mặc kệ nhiễu sóng
    esp_err_t result = esp_now_send(broadcast_addr, (uint8_t *)&tx_packet, sizeof(tx_packet));
    if (result != ESP_OK) {
      send_fail_count++;
    }
  }

  uint32_t now_ms = millis();
  if (now_ms - last_stats_ms >= 1000) {
    uint32_t elapsed_ms = now_ms - last_stats_ms;
    uint32_t seq_delta = tx_packet.seq - last_stats_seq;
    uint32_t send_rate_hz = (seq_delta * 1000UL) / max(elapsed_ms, 1UL);
    
    last_stats_ms = now_ms;
    last_stats_seq = tx_packet.seq;
    
    Serial.printf("STAT,seq,%lu,attempt,%lu,ok,%lu,fail,%lu,send_rate_hz,%lu,channel,%u\n",
                  (unsigned long)tx_packet.seq,
                  (unsigned long)send_attempt_count,
                  (unsigned long)send_ok_count,
                  (unsigned long)send_fail_count,
                  (unsigned long)send_rate_hz,
                  WIFI_CHANNEL);
  }
}