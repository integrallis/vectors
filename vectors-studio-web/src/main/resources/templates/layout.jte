@param String title
@param gg.jte.Content body
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${title} — Vectors Studio</title>
  <link rel="icon" type="image/png" sizes="32x32" href="/static/img/favicon-32x32.png">
  <link rel="icon" type="image/png" sizes="16x16" href="/static/img/favicon-16x16.png">
  <link rel="shortcut icon" href="/static/img/favicon.ico">
  <link rel="preconnect" href="https://rsms.me/">
  <link rel="stylesheet" href="https://rsms.me/inter/inter.css">
  <link rel="stylesheet" href="/static/studio.css">
  <script src="https://unpkg.com/htmx.org@1.9.12"></script>
  <script src="https://unpkg.com/alpinejs@3.x.x" defer></script>
</head>
<body>
  <header class="app-header">
    <a class="brand" href="/">
      <img src="/static/img/logo.png" alt="java-vectors logo">
      <span class="brand-name">Vectors <span class="brand-muted">Studio</span></span>
    </a>
    <nav class="app-nav">
      <a href="/collections">Collections</a>
    </nav>
  </header>
  <main>
    ${body}
  </main>
</body>
</html>
