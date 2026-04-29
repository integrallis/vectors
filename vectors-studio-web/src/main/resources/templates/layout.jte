@import java.util.Map
@param String title
@param gg.jte.Content body
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>${title} — Vectors Studio</title>
  <link rel="stylesheet" href="/static/studio.css">
  <script src="https://unpkg.com/htmx.org@1.9.12"></script>
  <script src="https://unpkg.com/alpinejs@3.x.x" defer></script>
</head>
<body>
  <header>
    <a href="/">Vectors Studio</a>
    <nav>
      <a href="/collections">Collections</a>
    </nav>
  </header>
  <main>
    ${body}
  </main>
</body>
</html>
