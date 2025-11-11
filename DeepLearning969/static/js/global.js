// 🌐 Global JS untuk semua halaman
document.addEventListener("DOMContentLoaded", () => {
    console.log("🚀 Website siap digunakan!");

    // Contoh animasi kecil untuk tombol
    const buttons = document.querySelectorAll("button");
    buttons.forEach(btn => {
        btn.addEventListener("mouseover", () => {
            btn.style.transform = "scale(1.05)";
        });
        btn.addEventListener("mouseout", () => {
            btn.style.transform = "scale(1)";
        });
    });
});
