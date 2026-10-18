# Test rehberi: yeni mimariyi simülasyonda ilk kez çalıştırmak

Bu rehber, sistemi **hiç çalıştırmamış** biri için yazıldı. Baştan sona,
sırayla gidersen sonunda şunları bilmiş olursun:

- sensörler ve kestirimci (`/odom/filtered`) doğru çalışıyor mu,
- ArduSub ile köprü (bridge) konuşuyor mu,
- her kanal aracı **doğru yöne** götürüyor mu,
- kontrolcü derinliğe, yöne ve konuma gidip orada durabiliyor mu,
- bir şey koptuğunda araç güvenli davranıyor mu.

Mimarinin neden böyle olduğunu merak edersen: [ARCHITECTURE.md](ARCHITECTURE.md).
Sensörlerin ve frame'lerin ne olduğunu bilmiyorsan önce
[ONBOARDING.md](ONBOARDING.md)'nin 1–4. bölümlerine göz at. Bu rehberi takip
etmek için ikisini de bitirmen şart değil.

> **Altın kural:** Aşamaları atlama. Her aşama bir öncekinin doğru olduğunu
> varsayar. Derinlik testi başarısız olduğunda sebebi çoğu zaman iki aşama
> önceki bir yön (işaret) hatasıdır.

---

## İçindekiler

0. [Ne lazım](#0-ne-lazım)
1. [Büyük resim: hangi terminalde ne çalışıyor](#1-büyük-resim)
2. [Aşama 0 — ROS'suz testler](#aşama-0--rossuz-testler)
3. [Aşama 1 — Build](#aşama-1--build)
4. [Aşama 2 — Sadece simülasyon ve kestirimci](#aşama-2--sadece-simülasyon-ve-kestirimci)
5. [Aşama 3 — ArduSub SITL ve köprü](#aşama-3--ardusub-sitl-ve-köprü)
6. [Aşama 4 — Kanal yönleri (kontrolcü kapalı)](#aşama-4--kanal-yönleri-kontrolcü-kapalı)
7. [Aşama 5 — Derinlik](#aşama-5--derinlik)
8. [Aşama 6 — Yön (yaw)](#aşama-6--yön-yaw)
9. [Aşama 7 — Yerinde durma (hold)](#aşama-7--yerinde-durma-hold)
10. [Aşama 8 — Konum hedefleri](#aşama-8--konum-hedefleri)
11. [Aşama 9 — Güvenlik davranışları](#aşama-9--güvenlik-davranışları)
12. [Kayıt almak](#kayıt-almak)
13. [Sorun giderme](#sorun-giderme)
14. [Sonuç formu](#sonuç-formu)

---

## 0. Ne lazım

- Ubuntu 22.04 + ROS 2 Humble
- Gazebo ve `ardupilot_gazebo` eklentisi, ArduPilot kaynak kodu
  (`sim_vehicle.py` için). Kurulum: ana [README.md](../README.md).
- Bu repo bir ROS 2 workspace'inin `src/` klasöründe, örneğin
  `~/nautronics_ws/src/nautronics_auv`.

Aşağıdaki komutlarda workspace kökü `~/nautronics_ws` olarak geçiyor. Seninki
farklıysa değiştir.

**Her yeni terminalde** önce şunu çalıştır, yoksa ROS komutları paketleri
bulamaz:

```bash
cd ~/nautronics_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
```

---

## 1. Büyük resim

Test sırasında aynı anda dört terminal açık olacak:

```
┌─ T1: Gazebo + bütün yığın ────────────┐   ┌─ T2: ArduSub SITL ─────────────┐
│ ros2 launch auv_bringup               │   │ sim_vehicle.py -v ArduSub ...  │
│   autonomy.launch.py sim:=true        │◀─▶│ (MAVProxy konsolu burada)      │
│   autopilot:=true                     │   │                                │
└───────────────────────────────────────┘   └────────────────────────────────┘
┌─ T3: Komut vermek ────────────────────┐   ┌─ T4: İzlemek ──────────────────┐
│ ros2 service call ...                 │   │ ros2 topic echo ...            │
│ ros2 topic pub ...                    │   │ ros2 run tf2_ros tf2_echo ...  │
└───────────────────────────────────────┘   └────────────────────────────────┘
```

Verinin yolu, aşağıdaki aşamalarda neyi test ettiğini bilmek için:

```
Gazebo sensörleri ─▶ gate'ler ─▶ kestirimci ─▶ /odom/filtered
                                                   │
                         hedef (sen, T3) ─▶ pose_controller ─▶ /pilot_demand/cmd_axis
                                                                    │
                                                              pilot_demand ─▶ /pilot_demand (PWM)
                                                                                   │
                                                  pixhawk_bridge2 ─▶ RC override ─▶ ArduSub ─▶ motorlar
```

Bazı kelimeler:

| Terim | Anlamı |
|---|---|
| **SITL** | "Software In The Loop": ArduSub'ın bilgisayarda çalışan hâli. Gerçek Pixhawk yerine geçer. |
| **ALT_HOLD** | ArduSub'ın derinlik tutma modu. Çubuklar ortadayken derinliği ve yönü kendisi tutar. |
| **RC override** | ArduSub'a "kumandanın çubukları şu konumda" demek. Her kanal 1100–1900 arası bir PWM, 1500 orta. |
| **odom** | Kestirimcinin dünya frame'i. Başlangıç noktası, kestirimcinin başladığı yer. **z yukarı pozitif**, yani 1 m derinlik `z = -1.0`. |
| **FLU** | Araç frame'i: x **ileri**, y **sol**, z **yukarı**. Pozitif yaw = **sola** (yukarıdan bakınca saat yönünün tersi) dönmek. |

---

## Aşama 0 — ROS'suz testler

Önce kodun kendisi doğru mu, ona bakarız. ROS bile gerekmez.

```bash
cd ~/nautronics_ws/src/nautronics_auv/auv_actuation && python3 -m pytest
cd ~/nautronics_ws/src/nautronics_auv/auv_core      && python3 -m pytest
cd ~/nautronics_ws/src/nautronics_auv/auv_sensors   && python3 -m pytest
```

✅ **Geçti:** Hepsi `passed`.
❌ Bir test kalıyorsa **dur**. Simülasyona geçme, önce o testi düzelt.

---

## Aşama 1 — Build

```bash
cd ~/nautronics_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

✅ **Geçti:** Sonda `Summary: N packages finished` yazıyor ve kırmızı `failed`
yok. Uyarılar (`stderr output`) sorun değil.

Yeni mesajlar oluştu mu, kontrol et:

```bash
ros2 interface show auv_core_msgs/msg/AxisDemand
ros2 interface show auv_core_msgs/msg/PilotDemand
```

✅ İkisi de alanlarını yazıyor (`surge`, `sway`, ... ve `pitch`, `roll`,
`throttle`, ...).

---

## Aşama 2 — Sadece simülasyon ve kestirimci

ArduSub olmadan sadece sensörleri ve kestirimciyi çalıştırıyoruz. Araç
hareket etmez, ama "nerede olduğunu" bilmesi gerekir.

**T1:**

```bash
ros2 launch auv_bringup autonomy.launch.py sim:=true
```

Gazebo açılır, araç havuzda 1 m derinlikte doğar. 10–20 saniye bekle.

**T4**, sırayla:

```bash
ros2 topic hz /dvl_gate/velocity     # ~8 Hz
ros2 topic hz /imu_gate/data         # ~50 Hz (simülasyonda)
ros2 topic hz /depth_gate/pose       # ~20 Hz
ros2 topic hz /odom/filtered         # ~30 Hz
```

Her birinde birkaç satır gördükten sonra `Ctrl+C` ile durdur.

```bash
ros2 topic echo /nav_status --once
```

✅ **Geçti:** `mode: 1`. Bu AIDED demek, yani DVL zemini görüyor. Kontrolcü
yatay eksenleri **sadece bu modda** sürer.

```bash
ros2 topic echo /imu_gate/health --once
```

✅ `status: 0` (OK). İlk 5–10 saniye `1` olabilir, gyro bias'ı ölçülüyor
demek. Bekle, tekrar bak.

```bash
ros2 run tf2_ros tf2_echo odom base_link
```

✅ Translation yaklaşık `[0, 0, -1.0]`, yani başlangıç noktası ve 1 m derinlik.
Değerler yavaşça kıpırdayabilir, ama birkaç dakikada santimetreler mertebesinde
kalmalı. Rotation satırındaki `RPY (degree)` değerleri roll ≈ 0, pitch ≈ 0.

❌ `/odom/filtered` yoksa veya `nav_status` `mode: 0`'da takılı kalıyorsa,
sorun kontrolde değil, sensör veya kestirimci tarafında.
[ONBOARDING.md §9](ONBOARDING.md#9-debugging-reading-the-health-topics)'a bak.
Bu aşama geçmeden ilerleme.

T1'i `Ctrl+C` ile kapat.

---

## Aşama 3 — ArduSub SITL ve köprü

### Başlatma sırası önemli

**T1** (önce Gazebo, çünkü SITL ona bağlanır):

```bash
ros2 launch auv_bringup autonomy.launch.py sim:=true autopilot:=true
```

**T2** (Gazebo açıldıktan sonra, ArduPilot klasöründe):

```bash
cd ~/ardupilot
sim_vehicle.py -v ArduSub -f JSON --model JSON --console
```

İlk seferde derleme birkaç dakika sürebilir. MAVProxy konsolu açılınca T1'de
şu satırı ara:

```
Pixhawk'a bağlanıldı! (command_mode: rc_override)
```

✅ **Geçti:** Bu satır var ve `rc_override` yazıyor.
❌ `legacy` yazıyorsa köprü eski modda başlamış demek. Eski launch dosyasını
kullanmışsındır, `autonomy.launch.py` kullan.
❌ Satır hiç gelmiyorsa köprü hâlâ ArduSub'ı bekliyor. Bkz.
[Sorun giderme](#sorun-giderme).

### ArduSub parametreleri

**T2'deki MAVProxy konsoluna** yaz (`STABILIZE>` gibi bir prompt görürsün):

```
param set RC_OVERRIDE_TIME 1
param set ARMING_CHECK 0
param show PILOT_SPEED_UP
param show PILOT_SPEED_DN
param show THR_DZ
```

- `RC_OVERRIDE_TIME 1`: bizden 1 saniye komut gelmezse ArduSub çubukları
  ortaya alır. Güvenlik için şart.
- `ARMING_CHECK 0`: **sadece simülasyonda**. Sahte sensörlerde arm kontrolleri
  takılabilir. Gerçek araçta bunu yapma.
- Son üç değeri bir kenara not et. Bunlar
  [pilot_demand_sim.yaml](../auv_actuation/config/pilot_demand_sim.yaml)'daki
  değerlerle aynı olmalı:

| ArduSub parametresi | Birimi | `pilot_demand_sim.yaml` karşılığı |
|---|---|---|
| `PILOT_SPEED_UP` | cm/s | `max_climb_rate_up` (m/s) = değer / 100 |
| `PILOT_SPEED_DN` | cm/s | `max_climb_rate_down` (m/s) = değer / 100. 0 ise UP ile aynı |
| `THR_DZ` | PWM | `throttle_deadzone_pwm` |

Farklıysa yaml'ı düzelt, T1'i kapatıp yeniden başlat. `--symlink-install` ile
build ettiysen yaml değişikliği için tekrar build gerekmez.

### Mod, çıkış, arm

**T3**, bu sırayla:

```bash
# 1. Derinlik tutma moduna al
ros2 service call /change_mode auv_interfaces/srv/SetVehicleMode "{mode_name: 'ALT_HOLD'}"

# 2. Çıkış katmanını aç. Şu an sadece "çubuklar ortada" gönderir.
ros2 service call /pilot_demand/enable std_srvs/srv/SetBool "{data: true}"

# 3. Arm et
ros2 service call /arm std_srvs/srv/SetBool "{data: true}"
```

Çıkışı arm'dan **önce** açıyoruz. Böylece ArduSub arm olduğu andan itibaren
sürekli "çubuklar ortada" bilgisi alır ve "pilot girdisi yok" uyarısına
düşmez.

✅ **Geçti:**
- MAVProxy prompt'u `ALT_HOLD>` oldu ve konsolda `ARMED` görünüyor.
- Araç Gazebo'da kendi derinliğinde **duruyor**: yüzeye çıkmıyor, batmıyor.

T4'te çıkışın ne gönderdiğine bak:

```bash
ros2 topic echo /pilot_demand
```

✅ Bütün kanallar `1500`.

---

## Aşama 4 — Kanal yönleri (kontrolcü kapalı)

**Bütün rehberin en önemli aşaması bu.** Kontrolcüyü devreye almadan, eksen
taleplerini elle verip aracın doğru yöne gittiğini görüyoruz. Burada bir yön
hatası varsa, kontrolcü aracı hedefinden **uzaklaştırır** ve sebebini ararken
saatler kaybedersin.

Kontrolcü (`pose_controller`) şu an kapalı, o yüzden çıkışa sadece senin
gönderdiğin talep gider. Komut `-r 20` ile saniyede 20 kez gönderilmeli.
`--once` ile gönderirsen yarım saniye sonra zaman aşımına düşer ve çubuklar
ortaya döner.

**T4'te** aracın hızını izle:

```bash
ros2 topic echo /odom/filtered --field twist.twist
```

Bu hızlar **aracın kendi frame'inde** (FLU): `linear.x` ileri, `linear.y`
sol, `linear.z` yukarı, `angular.z` sola dönüş.

Her satırı **T3**'te çalıştır, 3–5 saniye izle, `Ctrl+C` ile durdur, araç
durana kadar bekle, sonra sıradakine geç:

| # | Komut (T3) | `/pilot_demand`'da beklenen | Araç ne yapmalı | `twist` beklenen |
|---|---|---|---|---|
| a | `ros2 topic pub -r 20 /pilot_demand/cmd_axis auv_core_msgs/msg/AxisDemand "{surge: 0.3}"` | `forward: 1620` | **ileri** gider | `linear.x` **> 0** |
| b | `... "{surge: -0.3}"` | `forward: 1380` | **geri** gider | `linear.x` **< 0** |
| c | `... "{sway: 0.3}"` | `lateral: 1380` | **sola** kayar | `linear.y` **> 0** |
| d | `... "{climb_rate: 0.15}"` | `throttle` 1500'ün **üstünde** | **yukarı** çıkar | `linear.z` **> 0** |
| e | `... "{climb_rate: -0.15}"` | `throttle` 1500'ün **altında** | **aşağı** iner | `linear.z` **< 0** |
| f | `... "{yaw_rate: 0.3}"` | `yaw` 1500'ün **altında** | yukarıdan bakınca **sola** (saat yönünün tersine) döner | `angular.z` **> 0** |

(`...` yerine a satırındaki komutun başını yaz:
`ros2 topic pub -r 20 /pilot_demand/cmd_axis auv_core_msgs/msg/AxisDemand`.)

Dikkat: c'de PWM'in 1500'ün **altına** inmesi doğru. ArduSub'ın yan çubuğunda
yüksek değer sağ demek, bizde pozitif sway sol demek. f'de de aynı mantık:
ArduSub'da yüksek yaw saat yönü, bizde pozitif yaw saat yönünün tersi.

Ek kontroller:
- **d ve e:** `Ctrl+C`'den sonra araç **yeni derinliğinde kalmalı**. Eski
  derinliğine dönmemeli, yüzeye de çıkmamalı. ALT_HOLD tam olarak bunu yapar.
- **f:** `Ctrl+C`'den sonra dönme **durmalı** ve araç yeni yönünü korumalı.
- **Her biri:** `Ctrl+C`'den yaklaşık 0.5 saniye sonra T1'de
  `no axis demand ... sticks centred` uyarısı çıkmalı. Bu, zaman aşımı
  korumasının çalıştığını gösterir.

✅ **Geçti:** Altı satırın altısında da araç tablodaki yöne gitti.

❌ **Bir eksen ters gidiyorsa:** Düzeltme **sadece**
[demand_mapping.py](../auv_actuation/auv_actuation/demand_mapping.py)'de
yapılır. Kontrolcüde işaret çevirme, mimari kuralı bu. Düzelttikten sonra
[test_demand_mapping.py](../auv_actuation/test/test_demand_mapping.py)'deki
ilgili testi de güncelle, yoksa hata bir gün geri gelir.

---

## Aşama 5 — Derinlik

Artık kontrolcüyü açıyoruz.

**T3:**

```bash
ros2 service call /pose_controller/enable std_srvs/srv/SetBool "{data: true}"
```

T1'de `holding at (x, y, z)` görürsün. Kontrolcü açıldığı an aracı **olduğu
yerde** tutmaya başlar. Araç yerinden oynamamalı.

**T4:** Derinlik hatasını izle:

```bash
ros2 topic echo /pose_controller/pose_error --field pose.position.z
```

**T3:** Mevcut derinlikten 1 m aşağıyı hedef ver. Örneğin araç `z = -1.0`'da
ise hedef `-2.0`:

```bash
ros2 topic pub --once /pose_controller/target_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: odom}, pose: {position: {x: 0.0, y: 0.0, z: -2.0}, orientation: {w: 0.0}}}"
```

> ⚠️ **`orientation: {w: 0.0}` unutma.** ROS 2'de orientation verilmezse
> varsayılan `w: 1` olur ve bu "yaw = 0'a dön" demektir. `w: 0.0` (sıfır
> quaternion) ise "yönüne karışma" demektir.
>
> `x: 0.0, y: 0.0` da bir hedeftir: araç odom'un başlangıç noktasına gitmeye
> çalışır. Bu aşamada araç zaten oraya yakın olduğu için sorun değil. Değilse
> `tf2_echo odom base_link`'ten okuduğun mevcut x ve y'yi yaz.

✅ **Geçti (kabaca beklenen, kesin şartname değil):**
- `pose_error` z değeri **-1.0'dan 0'a** doğru küçülür.
- Araç 5–10 saniyede yeni derinliğe varır.
- 10 cm'den fazla aşmaz (overshoot), salınım yapmaz.
- `/pose_controller/at_target` `true` olur.

Geri dön (`z: -1.0` ile aynı komut) ve yukarı yönde de dene.

❌ Hiç gitmiyorsa veya yavaşça kayıyorsa: `THR_DZ` ile
`throttle_deadzone_pwm` uyuşmuyor olabilir (Aşama 3).
❌ Aşıp geri dönüyor ve salınıyorsa:
[pose_controller.yaml](../auv_actuation/config/pose_controller.yaml)'da
`kp_depth`'i düşür (örneğin 0.8 → 0.5).

---

## Aşama 6 — Yön (yaw)

**T4:** Mevcut yönü izle:

```bash
ros2 run tf2_ros tf2_echo odom base_link
```

`Rotation: in RPY (degree)` satırındaki **üçüncü** sayı yaw'dır.

**T3:** Aşağıdaki tablodan bir yön seç. Quaternion'ı elle hesaplamana gerek
yok:

| İstenen yaw | `orientation` |
|---|---|
| 0° | `{z: 0.0, w: 1.0}` |
| +45° (sola) | `{z: 0.3827, w: 0.9239}` |
| +90° (sola) | `{z: 0.7071, w: 0.7071}` |
| 180° | `{z: 1.0, w: 0.0}` |
| -90° (sağa) | `{z: -0.7071, w: 0.7071}` |

Konumu ve derinliği mevcut değerlerde tutup sadece yönü değiştir. Örneğin araç
`(0, 0, -1.0)`'da ise:

```bash
ros2 topic pub --once /pose_controller/target_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: odom}, pose: {position: {x: 0.0, y: 0.0, z: -1.0}, orientation: {z: 0.7071, w: 0.7071}}}"
```

✅ **Geçti:**
- Yaw yaklaşık 90°'ye gelir ve orada durur.
- **Sola** döner. Pozitif yaw her zaman sola dönüş demek.
- 5° fazlasını aşmaz, 5–10 saniyede varır.
- Dönerken derinlik değişmez.

Sonra 180° ve -90° dene. **Kısa yoldan** dönmeli: +90°'den -90°'ye giderken
hangi yöne döndüğü önemli değil, ama 170°'den -170°'ye giderken 340° değil
20° dönmeli.

❌ Salınıyorsa `kp_yaw`'ı düşür.
❌ Hiç dönmüyorsa veya ters dönüyorsa Aşama 4'ün f satırına geri dön.

---

## Aşama 7 — Yerinde durma (hold)

Kendi POS_HOLD'umuzun testi.

**T3:**

```bash
ros2 service call /pose_controller/hold std_srvs/srv/Trigger
```

Bu komut aracı **şu anki** konumunda, derinliğinde ve yönünde kilitler.

**Test 1 — kayma:** 2 dakika bekle. T4'te `tf2_echo odom base_link` ile x, y,
z ve yaw'ı izle.
✅ Hepsi başladığı yerin ±10 cm / ±3° içinde kalıyor.

**Test 2 — itme:** Gazebo arayüzünde **Apply Force/Torque** eklentisini aç
(sağ üstteki ⋮ menüsü → eklenti listesinde arat), aracı seç ve bir saniyeliğine
yandan kuvvet uygula.
✅ Araç itildikten sonra kendi kendine eski yerine **geri döner** ve orada
durur.

❌ Geri dönüyor ama sürekli biraz yanında duruyorsa: `surge`/`sway` altındaki
`ki` çok düşük.
❌ Etrafında salınıyorsa: `surge`/`sway` altındaki `kp` çok yüksek.

---

## Aşama 8 — Konum hedefleri

Havuz 25×50 m, ama aracın odom yönünün havuzun hangi yönüne denk geldiğini
bilmiyoruz. İlk denemelerde **3 m'yi geçme** ve Gazebo'yu izle.

**T4:**

```bash
ros2 topic echo /pose_controller/pose_error --field pose.position
```

**T3**, sırayla (yönü korumak için `w: 0.0`):

```bash
# 3 m ileri (odom x)
ros2 topic pub --once /pose_controller/target_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: odom}, pose: {position: {x: 3.0, y: 0.0, z: -1.0}, orientation: {w: 0.0}}}"

# çapraz
ros2 topic pub --once /pose_controller/target_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: odom}, pose: {position: {x: 1.0, y: 2.0, z: -1.5}, orientation: {w: 0.0}}}"

# başlangıca dön, 0° yönüne bakarak
ros2 topic pub --once /pose_controller/target_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: odom}, pose: {position: {x: 0.0, y: 0.0, z: -1.0}, orientation: {z: 0.0, w: 1.0}}}"
```

✅ **Geçti:**
- Araç hedefe **yaklaşık 0.5 m/s'yi aşmadan** gider, yaklaştıkça yavaşlar.
- Hedefin 10 cm yakınında durur ve `at_target` `true` olur.
- Çapraz hedefte derinlik ve konum aynı anda değişir.

### Kestirimci de doğru mu?

Kontrolcü, kestirimcinin "buradasın" dediği yere gider. Kestirimci yanılıyorsa
araç da yanlış yere gider, ama `pose_error` sıfır görünür. Bunu yakalamak için
Gazebo'nun **gerçek** konumuyla karşılaştır:

```bash
ros2 topic echo /ground_truth/odom --field pose.pose.position
```

Gerçek konumun orijini odom'unkinden farklı. Araç Gazebo'da `(-9, 5, -1)`'de
doğar, odom ise `(0, 0, -1)`'den başlar. Bu yüzden **mutlak değerleri değil,
değişimi** karşılaştır: "3 m ileri" komutundan önce ve sonra iki konumu da not
et.

✅ İki değişim arasındaki fark 3 m'lik bir harekette 10–15 cm'den az.

---

## Aşama 9 — Güvenlik davranışları

Araç hold'dayken her birini tek tek dene:

| # | Ne yap | Beklenen |
|---|---|---|
| a | T1'de çalışan `pose_controller`'ı öldür: yeni bir terminalde `pkill -f pose_controller` | ~0.5 s sonra T1'de `sticks centred` görünür. Araç derinliğini ve yönünü korur, yatayda yavaşça durur. |
| b | `ros2 service call /pilot_demand/enable std_srvs/srv/SetBool "{data: false}"` | `/pilot_demand` yayını kesilir. ~1 s sonra (`RC_OVERRIDE_TIME`) ArduSub çubukları ortalar. Araç derinliğini korur. |
| c | `ros2 service call /pose_controller/enable std_srvs/srv/SetBool "{data: false}"` ve sonra tekrar `true` | Tekrar açıldığında araç **eski hedefe gitmez**, olduğu yerde durur (`holding at ...`). |

a'dan sonra yığını baştan başlatman gerekir (T1'de `Ctrl+C`, sonra launch).

✅ Üçünde de araç kontrolsüz hareket etmedi.

---

## Kayıt almak

Bir şey ters gittiğinde ve birine sormak istediğinde, kayıt olmadan kimse
sana yardım edemez. Her test oturumunda bir bag kaydı al:

```bash
ros2 bag record -o test_$(date +%Y%m%d_%H%M) \
  /odom/filtered /ground_truth/odom /nav_status \
  /pose_controller/pose_error /pose_controller/desired_velocity /pose_controller/at_target \
  /pilot_demand/cmd_axis /pilot_demand \
  /dvl_gate/health /imu_gate/health /depth_gate/health
```

Oluşan klasörü (`test_...`) sakla. Sonradan PlotJuggler ile açıp grafiklere
bakabilirsin (`sudo apt install ros-humble-plotjuggler-ros`, sonra
`ros2 run plotjuggler plotjuggler`).

---

## Sorun giderme

| Belirti | Muhtemel sebep | Ne yap |
|---|---|---|
| T1'de `Pixhawk'a bağlanıldı` hiç gelmiyor | SITL çalışmıyor, ya da MAVProxy `127.0.0.1:14551`'e çıkış vermiyor | T2'de `sim_vehicle.py` açık mı bak. MAVProxy varsayılan olarak 14550 ve 14551'e çıkış verir. Kendi parametreleriyle başlattıysan `--out=udp:127.0.0.1:14551` ekle. |
| SITL açılıyor ama `No JSON sensor message received` diyor | Gazebo SITL'den sonra açıldı, ya da Gazebo duraklatılmış | Önce T1, sonra T2. Gazebo'nun oynat (▶) tuşu basılı olmalı. |
| `/arm` başarısız | Arm kontrolleri takıldı | Simülasyonda `param set ARMING_CHECK 0`. T2'de hata mesajını oku. |
| Arm oldu ama araç hiç tepki vermiyor | Mod ALT_HOLD değil, çıkış kapalı, ya da köprü `legacy` modda | MAVProxy prompt'u `ALT_HOLD>` mı? `ros2 topic echo /pilot_demand` bir şey basıyor mu? T1'de `command_mode: rc_override` yazıyor mu? |
| `/pilot_demand` yayınlanıyor ama ArduSub kanalları görmüyor | RC override kabul edilmiyor | MAVProxy'de `param show SYSID_MYGCS`: 255 olmalı. `param show RC_OVERRIDE_TIME`: 1 olmalı. |
| Arm'dan bir süre sonra kendiliğinden disarm oluyor | Pilot girdisi failsafe'i | Çıkışı (`/pilot_demand/enable`) arm'dan **önce** aç. `param show FS_PILOT_INPUT` ile ayarı kontrol et. |
| Derinlik ve yaw çalışıyor, araç yatayda hiç gitmiyor | DVL zemini görmüyor (`nav_status` AIDED değil) | `ros2 topic echo /nav_status --once`. T1'de `horizontal axes centred` uyarısı var mı? |
| T1'de `no odometry; not commanding` | Kestirimci çalışmıyor | Aşama 2'ye dön. |
| T1'de `target_pose in frame '' ignored` | Hedefte `frame_id` eksik | Komuta `header: {frame_id: odom}` ekle. |
| Hedef verince araç önce bir yöne dönüyor | `orientation` verilmedi, varsayılan `w: 1` yaw = 0 demek | Yönü korumak için `orientation: {w: 0.0}`. |
| Bir eksen ters gidiyor | Yön eşlemesi hatalı | Aşama 4. Düzeltme sadece `demand_mapping.py`'de yapılır. |
| T1'de `demand beyond full stick, clipped` | Kontrolcü ArduSub'ın verebileceğinden hızlı bir şey istiyor | `pose_controller.yaml`'daki `max_climb_rate` / `max_yaw_rate`, `pilot_demand_sim.yaml`'daki karşılıklarından küçük olmalı. |
| Hedefe varıyor ama etrafında salınıyor | Kazanç yüksek | İlgili `kp`'yi yarıya indir. Sıra: önce derinlik ve yaw, sonra surge/sway, en son `kp_position`. |
| Hedefin yanında duruyor, tam varamıyor | Yatayda integral az, ya da deadzone | `surge`/`sway` `ki`'sini biraz artır. |

---

## Sonuç formu

Testi bitirince bunu doldur ve commit mesajına ya da bir issue'ya ekle.
Sonraki kişi nerede kaldığını bilsin.

```
Tarih:
Kim:
Commit (git log --oneline -1):

Aşama 0  ROS'suz testler           [ ] geçti  [ ] kaldı   not:
Aşama 1  Build                     [ ] geçti  [ ] kaldı   not:
Aşama 2  Sensörler + kestirimci    [ ] geçti  [ ] kaldı   not:
Aşama 3  SITL + köprü              [ ] geçti  [ ] kaldı   not:
         PILOT_SPEED_UP= ___  PILOT_SPEED_DN= ___  THR_DZ= ___
Aşama 4  Kanal yönleri             a[ ] b[ ] c[ ] d[ ] e[ ] f[ ]
Aşama 5  Derinlik                  [ ] geçti  [ ] kaldı   varış süresi: ___ s  aşma: ___ cm
Aşama 6  Yaw                       [ ] geçti  [ ] kaldı   varış süresi: ___ s  aşma: ___ °
Aşama 7  Hold                      kayma: ___ cm / 2 dk   itme sonrası geri döndü [ ]
Aşama 8  Konum                     [ ] geçti  [ ] kaldı   kestirimci hatası: ___ cm / 3 m
Aşama 9  Güvenlik                  a[ ] b[ ] c[ ]

Değiştirilen parametreler (dosya: eski → yeni):

Bag kaydı:
```
